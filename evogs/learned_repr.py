"""EvoGS learned representation: trainable collinear (ψ, α) split children.

This is the *faithful* EvoGS parametrization (vs the skeleton's two free
children + post-hoc ψ/α extraction). Each split pair shares a trainable
refinement direction ψ and a per-attribute asymmetry α against a frozen
parent snapshot:

    C1 = base + ψ
    C2 = base − α ⊙ ψ            (collinear: both children on the line base±ψ)

with α ∈ R^A, A=5 attribute groups (pos, rot, scale, opacity, SH). Only the
frontier split pairs' (ψ, α) are trainable; the parent snapshot `base` is
frozen (detached). Inherited level-start leaves have ψ=0; whether their ψ may
train is controlled by `freeze_inherited` (strict paper reading = True).

Storage model (paper §5): root P_root + per-split (ψ ∈ R^D, α ∈ R^5). Because
ψ and α are *stored parameters* here, residual extraction is exact (no
least-squares back-solve as in the skeleton's evolution_tree.py).

The class owns its own Adam optimizer (rebuilt on each topology change, which
only happens at the ~5 split events per level) and exposes a reconstruct()
that returns a plain dict suitable for `rasterize_splats`.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat


# Parameter groups and their attribute index for α (A=5: pos, rot, scale, opa, sh)
GROUPS = ["means", "quats", "scales", "opacities", "sh0", "shN"]
ATTR_IDX = {"means": 0, "quats": 1, "scales": 2, "opacities": 3, "sh0": 4, "shN": 4}
N_ATTR = 5

# softplus(LOG_ALPHA_ONE) == 1.0  →  init so that α ≈ 1 (symmetric) at split time
LOG_ALPHA_ONE = math.log(math.e - 1.0)
_LOG2 = 0.6931471805599453
_SHRINK = 1.6


def _sign_view(sign: Tensor, ref: Tensor) -> Tensor:
    """Reshape per-leaf [N] sign to broadcast against a group tensor `ref`."""
    return sign.view(ref.shape[0], *([1] * (ref.dim() - 1)))


class LearnedLeafSet:
    """Active leaf set parametrized as frozen base + trainable collinear (ψ, α)."""

    def __init__(
        self,
        init_splats: dict,
        device: str,
        asymmetric: bool = True,
        freeze_inherited: bool = True,
    ) -> None:
        self.device = device
        self.asymmetric = asymmetric
        self.freeze_inherited = freeze_inherited

        N = init_splats["means"].shape[0]
        # Frozen parent snapshot (base). For level-start leaves base = loaded value.
        self.base: Dict[str, Tensor] = {
            g: init_splats[g].detach().clone().to(device).contiguous() for g in GROUPS
        }
        # Trainable per-leaf ψ (init 0). C2 leaves do not use their own ψ row.
        self.psi = nn.ParameterDict(
            {g: nn.Parameter(torch.zeros_like(self.base[g])) for g in GROUPS}
        )
        # Trainable per-leaf log-α (only C2 rows are used). init α≈1.
        self.log_alpha = nn.Parameter(
            torch.full((N, N_ATTR), LOG_ALPHA_ONE, device=device)
        )

        # Topology bookkeeping (positional, rebuilt on every change)
        self.branch = torch.zeros(N, dtype=torch.long, device=device)   # 0=C1/inherited, 1=C2
        self.pair_id = torch.full((N,), -1, dtype=torch.long, device=device)
        self.leaf_id = torch.arange(N, dtype=torch.long, device=device)
        self.psi_src = torch.arange(N, dtype=torch.long, device=device)  # ψ source row per leaf

        self._next_leaf_id = N
        self._next_pair_id = 0
        self.splits: List[dict] = []  # {parent_id, c1_id, c2_id, level}

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._lr_spec: Optional[dict] = None
        self._rebuild_indices()

    # ------------------------------------------------------------------ utils
    @property
    def N(self) -> int:
        return self.base["means"].shape[0]

    def _rebuild_indices(self) -> None:
        """Recompute psi_src (sibling-sharing of ψ) after a topology change."""
        N = self.N
        is_c1 = (self.branch == 0) & (self.pair_id >= 0)
        is_c2 = self.branch == 1
        psi_src = torch.arange(N, device=self.device)
        if is_c2.any():
            maxp = int(self.pair_id.max().item()) + 1
            c1_pos_by_pair = torch.full((maxp,), -1, dtype=torch.long, device=self.device)
            c1_pos_by_pair[self.pair_id[is_c1]] = torch.arange(N, device=self.device)[is_c1]
            src = c1_pos_by_pair[self.pair_id[is_c2]]
            # Fallback: any C2 whose C1 vanished points to itself (treated standalone)
            src = torch.where(src >= 0, src, torch.arange(N, device=self.device)[is_c2])
            psi_src[is_c2] = src
        self.psi_src = psi_src
        # Per-leaf trainability mask (for strict frozen-inherited mode)
        if self.freeze_inherited:
            self.trainable = self.pair_id >= 0
        else:
            self.trainable = torch.ones(N, dtype=torch.bool, device=self.device)

    def alpha(self) -> Tensor:
        if self.asymmetric:
            return F.softplus(self.log_alpha)
        return torch.ones_like(self.log_alpha)

    # ------------------------------------------------------------ reconstruct
    def reconstruct(self) -> Dict[str, Tensor]:
        """Differentiable per-leaf params for rasterization: base + sign·ψ."""
        alpha = self.alpha()                       # [N, 5]
        is_c2 = self.branch == 1                    # [N]
        out: Dict[str, Tensor] = {}
        for g in GROUPS:
            a = ATTR_IDX[g]
            psi_eff = self.psi[g][self.psi_src]    # share ψ from sibling C1 for C2
            sign = torch.where(is_c2, -alpha[:, a], torch.ones_like(alpha[:, a]))  # [N]
            out[g] = self.base[g] + _sign_view(sign, psi_eff) * psi_eff
        out["quats"] = F.normalize(out["quats"], dim=-1)
        return out

    @torch.no_grad()
    def materialize(self) -> Dict[str, Tensor]:
        return {k: v.detach() for k, v in self.reconstruct().items()}

    # ------------------------------------------------------------------ split
    @torch.no_grad()
    def split(self, sel_mask: Tensor, level: int) -> None:
        """Split selected frontier leaves into collinear (ψ, α) child pairs.

        Children share a `base` = standard split-init transform of the parent's
        current reconstructed value (means copied, scale ÷1.6, opacity ÷2), then
        diverge along ψ (means seeded rotation-aligned; other groups ψ=0).
        """
        sel = torch.where(sel_mask)[0]
        keep = torch.where(~sel_mask)[0]
        n = len(sel)
        if n == 0:
            return

        recon = self.reconstruct()  # current values (detached below)
        parent_ids = self.leaf_id[sel].clone()

        # ---- child base (shared by C1 & C2): split-init transform of parent ----
        p_scales = torch.exp(recon["scales"][sel].detach())          # [n,3]
        p_quats = F.normalize(recon["quats"][sel].detach(), dim=-1)  # [n,4]
        R = normalized_quat_to_rotmat(p_quats)                       # [n,3,3]
        dirs = F.normalize(torch.randn(n, 3, device=self.device), dim=-1)
        psi_means = torch.einsum("nij,nj,nj->ni", R, p_scales, dirs)  # [n,3]

        child_base: Dict[str, Tensor] = {}
        for g in GROUPS:
            v = recon[g][sel].detach()
            if g == "scales":
                v = torch.log(torch.exp(v) / _SHRINK)
            elif g == "opacities":
                v = v - _LOG2
            child_base[g] = v  # [n, ...]

        child_psi: Dict[str, Tensor] = {
            g: torch.zeros((n, *self.base[g].shape[1:]), device=self.device)
            for g in GROUPS
        }
        child_psi["means"] = psi_means

        # ---- assign ids / pairs (interleaved C1_i, C2_i) ----
        new_leaf_ids = torch.arange(
            self._next_leaf_id, self._next_leaf_id + 2 * n, device=self.device
        )
        self._next_leaf_id += 2 * n
        new_pairs = torch.arange(
            self._next_pair_id, self._next_pair_id + n, device=self.device
        )
        self._next_pair_id += n

        c1_ids = new_leaf_ids[0::2]
        c2_ids = new_leaf_ids[1::2]
        for i in range(n):
            self.splits.append({
                "parent_id": int(parent_ids[i].item()),
                "c1_id": int(c1_ids[i].item()),
                "c2_id": int(c2_ids[i].item()),
                "level": level,
            })

        # ---- rebuild tensors: [kept | C1s | C2s] ----
        def cat_kept_children(kept: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
            return torch.cat([kept, c1, c2], dim=0)

        for g in GROUPS:
            base_g = cat_kept_children(self.base[g][keep], child_base[g], child_base[g])
            psi_g = cat_kept_children(
                self.psi[g][keep], child_psi[g],
                torch.zeros_like(child_psi[g]),  # C2 uses sibling ψ
            )
            self.base[g] = base_g.contiguous()
            self.psi[g] = nn.Parameter(psi_g.contiguous())

        la = torch.full((n, N_ATTR), LOG_ALPHA_ONE, device=self.device)
        self.log_alpha = nn.Parameter(
            torch.cat([self.log_alpha[keep], la, la.clone()], dim=0).contiguous()
        )

        zeros_n = torch.zeros(n, dtype=torch.long, device=self.device)
        ones_n = torch.ones(n, dtype=torch.long, device=self.device)
        self.branch = torch.cat([self.branch[keep], zeros_n, ones_n])
        self.pair_id = torch.cat([self.pair_id[keep], new_pairs, new_pairs])
        self.leaf_id = torch.cat([self.leaf_id[keep], c1_ids, c2_ids])

        self._rebuild_indices()
        self._build_optimizer()  # tensors replaced → fresh optimizer

    # ------------------------------------------------------------------ prune
    @torch.no_grad()
    def prune(self, prune_mask: Tensor) -> int:
        """Remove leaves where prune_mask is True. Surviving C2 whose C1 is
        pruned is baked into a standalone leaf (base = current value, ψ=0)."""
        if not prune_mask.any():
            return 0
        keep = ~prune_mask
        # Bake survivors whose ψ-source is being pruned (orphaned C2)
        src_pruned = prune_mask[self.psi_src]
        orphan = keep & src_pruned & (self.branch == 1)
        if orphan.any():
            recon = self.reconstruct()
            for g in GROUPS:
                self.base[g][orphan] = recon[g][orphan].detach()
                self.psi[g].data[orphan] = 0.0
            self.branch[orphan] = 0
            self.pair_id[orphan] = -1

        for g in GROUPS:
            self.base[g] = self.base[g][keep].contiguous()
            self.psi[g] = nn.Parameter(self.psi[g][keep].contiguous())
        self.log_alpha = nn.Parameter(self.log_alpha[keep].contiguous())
        self.branch = self.branch[keep]
        self.pair_id = self.pair_id[keep]
        self.leaf_id = self.leaf_id[keep]

        self._rebuild_indices()
        self._build_optimizer()
        return int(prune_mask.sum().item())

    # -------------------------------------------------------------- optimizer
    def set_lr_spec(self, lr_spec: dict) -> None:
        """lr_spec maps {group_name: lr} for ψ groups + {'log_alpha': lr}."""
        self._lr_spec = lr_spec
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        if self._lr_spec is None:
            return
        BS = 1
        eps = 1e-15
        betas = (0.9, 0.999)
        groups = []
        for g in GROUPS:
            groups.append({"params": [self.psi[g]], "lr": self._lr_spec[g], "name": g})
        if self.asymmetric:
            groups.append({
                "params": [self.log_alpha],
                "lr": self._lr_spec["log_alpha"], "name": "log_alpha",
            })
        self.optimizer = torch.optim.Adam(groups, eps=eps, betas=betas)

    def scale_means_lr(self, factor: float) -> None:
        """Apply LR decay to the ψ-means group (mirrors skeleton means schedule)."""
        if self.optimizer is None or self._lr_spec is None:
            return
        for pg in self.optimizer.param_groups:
            if pg.get("name") == "means":
                pg["lr"] = self._lr_spec["means"] * factor

    def mask_grads(self) -> None:
        """Zero ψ grads on frozen/non-owning rows; zero log_alpha grads on C1."""
        tr = self.trainable
        for g in GROUPS:
            if self.psi[g].grad is not None:
                # only C1/inherited rows own ψ; C2 ψ rows are unused
                own = (self.branch == 0) & tr
                self.psi[g].grad *= _sign_view(own.float(), self.psi[g].grad)
        if self.asymmetric and self.log_alpha.grad is not None:
            is_c2 = (self.branch == 1).float().unsqueeze(-1)
            self.log_alpha.grad *= is_c2

    def step(self) -> None:
        self.mask_grads()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # -------------------------------------------------------------- serialize
    @torch.no_grad()
    def save(self, tree_dir: str) -> dict:
        """Write topology.json + residuals.npz (exact ψ/α per surviving pair)."""
        os.makedirs(tree_dir, exist_ok=True)
        topology = {
            "n_total_ids": self._next_leaf_id,
            "n_pairs": self._next_pair_id,
            "n_splits": len(self.splits),
            "asymmetric": self.asymmetric,
            "freeze_inherited": self.freeze_inherited,
            "splits": self.splits,
        }
        with open(os.path.join(tree_dir, "topology.json"), "w") as f:
            json.dump(topology, f)

        # Exact residuals: for each surviving pair, ψ (flat over groups) + α[5]
        is_c1 = (self.branch == 0) & (self.pair_id >= 0)
        c1_pos = torch.where(is_c1)[0]
        psi_rows, alpha_rows = [], []
        alpha = self.alpha()
        # map pair -> c2 pos for alpha
        is_c2 = self.branch == 1
        c2_pos_all = torch.where(is_c2)[0]
        pair_to_c2 = {int(self.pair_id[p].item()): int(p.item()) for p in c2_pos_all}
        for p in c1_pos:
            pid = int(self.pair_id[p].item())
            psi_flat = torch.cat([self.psi[g][p].flatten() for g in GROUPS])
            psi_rows.append(psi_flat.cpu().numpy())
            c2p = pair_to_c2.get(pid)
            a = alpha[c2p].cpu().numpy() if c2p is not None else np.ones(N_ATTR, np.float32)
            alpha_rows.append(a.astype(np.float32))
        info = {"n_residual_pairs": len(psi_rows)}
        if psi_rows:
            np.savez_compressed(
                os.path.join(tree_dir, "residuals.npz"),
                psi=np.stack(psi_rows).astype(np.float32),
                alpha=np.stack(alpha_rows).astype(np.float32),
            )
            info["psi_shape"] = list(np.stack(psi_rows).shape)
        return info

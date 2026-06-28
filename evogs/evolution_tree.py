"""EvoGS Evolution Tree: tracks binary split history for scalable 3DGS streaming.

Each leaf is a Gaussian primitive. When a leaf is refined, it becomes an
internal node and two children take its place in the active rasterization set.

Children are parameterized relative to their parent:
    C1 = P + ψ
    C2 = P - α ⊙ ψ

ψ ∈ R^D (refinement direction, same dims as parent param vector)
α ∈ R^5 (one asymmetry scalar per parameter group: pos, quat, scale, opa, sh)

During training, leaf params are stored as dense tensors (no explicit ψ/α).
Residuals are extracted post-training for compression evaluation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat


# Parameter group names (order matters for α indexing)
_PARAM_GROUPS = ["means", "quats", "scales", "opacities", "sh0", "shN"]


@dataclass
class SplitRecord:
    parent_leaf_id: int
    child1_leaf_id: int
    child2_leaf_id: int
    level: int


class EvolutionTree:
    """Tracks the binary split history for EvoGS.

    During training, leaf parameters live in a standard ParameterDict (dense
    tensors). This class records the tree topology and parent snapshots so that
    post-hoc residual extraction and compression evaluation are possible.
    """

    def __init__(self) -> None:
        self.splits: List[SplitRecord] = []
        self._next_id: int = 0
        # child leaf_id → numpy dict of parent params at time of split
        self._child_to_parent: Dict[int, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_roots(self, n: int) -> Tensor:
        """Assign leaf IDs to the initial leaves. Returns a leaf_ids LongTensor."""
        ids = torch.arange(n, dtype=torch.long)
        self._next_id = n
        return ids

    # ------------------------------------------------------------------
    # Evolution split
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evo_split(
        self,
        mask: Tensor,
        splats: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        leaf_ids: Tensor,
        level: int,
    ) -> Tensor:
        """Perform EvoGS symmetric split on leaves selected by mask.

        Modifies splats and optimizers in-place. Does NOT touch the gradient
        accumulation state (caller resets it after split+prune).

        Returns updated leaf_ids tensor aligned with the new splats layout:
            [surviving leaves | child1_of_sel[0] | child2_of_sel[0] | ...]
        """
        from gsplat.strategy.ops import _update_param_with_optimizer

        device = mask.device
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]
        n_sel = len(sel)

        if n_sel == 0:
            return leaf_ids

        # Snapshot parent params before any modification (CPU, float32, always ≥1-D)
        parent_cpu: Dict[str, np.ndarray] = {
            k: splats[k][sel].detach().cpu().float().numpy() for k in splats
        }
        parent_leaf_ids_sel = leaf_ids[sel]
        # Note: parent_cpu[k] has shape [n_sel, ...]. Indexing with scalar i gives
        # a scalar (numpy.float32) for 1-D params (e.g. opacities). We use atleast_1d
        # when storing per-parent snapshots to guarantee ndarray types.

        # Compute per-parent perturbation ψ for means
        # One shared direction per parent (symmetric: C2 = P - ψ, i.e. α = 1 init)
        scales = torch.exp(splats["scales"][sel])          # [n_sel, 3]
        quats = F.normalize(splats["quats"][sel], dim=-1)  # [n_sel, 4]
        rotmats = normalized_quat_to_rotmat(quats)         # [n_sel, 3, 3]
        # Random unit direction, rotated and scaled by parent extent
        dirs = F.normalize(torch.randn(n_sel, 3, device=device), dim=-1)
        psi_means = torch.einsum("nij,nj,nj->ni", rotmats, scales, dirs)  # [n_sel, 3]

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "means":
                c1 = p[sel] + psi_means
                c2 = p[sel] - psi_means
                p_split = torch.cat([c1, c2], dim=0)  # interleave below
            elif name == "scales":
                shrunken = torch.log(scales / 1.6)
                p_split = shrunken.repeat(2, 1)
            elif name == "quats":
                normed = F.normalize(p[sel], dim=-1)
                p_split = normed.repeat(2, 1)
            elif name == "opacities":
                # logit(sigmoid(P) × 0.5) = P - log(2)
                halved = p[sel] - 0.6931471805599453
                p_split = halved.repeat(1) if halved.dim() == 1 else halved.repeat(2, *([1] * (halved.dim() - 1)))
                # ensure correct shape for 1-D opacities
                if halved.dim() == 1:
                    p_split = torch.cat([halved, halved], dim=0)
            else:
                # sh0, shN: copy appearance to both children
                repeats = [2] + [1] * (p.dim() - 1)
                p_split = p[sel].repeat(repeats)

            # Layout: [rest, C1_0, C1_1, ..., C2_0, C2_1, ...]
            # We interleave per-parent: [rest, C1_0, C2_0, C1_1, C2_1, ...]
            # Actually ops.split() puts all rest first then all children. Let's
            # do the same but ordered as [C1_0, C2_0, C1_1, C2_1, ...].
            if name == "means":
                # Already [c1_0..c1_{n-1}, c2_0..c2_{n-1}]; interleave
                c1_block = p_split[:n_sel]   # [n_sel, 3]
                c2_block = p_split[n_sel:]   # [n_sel, 3]
                interleaved = torch.stack([c1_block, c2_block], dim=1).reshape(2 * n_sel, 3)
                p_new = torch.cat([p[rest], interleaved])
            elif name in ("scales", "quats"):
                c1_block = p_split[:n_sel]
                c2_block = p_split[n_sel:]
                interleaved = torch.stack([c1_block, c2_block], dim=1).reshape(2 * n_sel, p.shape[-1])
                p_new = torch.cat([p[rest], interleaved])
            elif name == "opacities":
                c1_block = p_split[:n_sel]
                c2_block = p_split[n_sel:]
                interleaved = torch.stack([c1_block, c2_block], dim=1).reshape(2 * n_sel)
                p_new = torch.cat([p[rest], interleaved])
            else:
                # sh0, shN: [n_sel * 2, K, 3] — interleave
                c1_block = p_split[:n_sel]
                c2_block = p_split[n_sel:]
                interleaved = torch.stack([c1_block, c2_block], dim=1).reshape(
                    2 * n_sel, *p.shape[1:]
                )
                p_new = torch.cat([p[rest], interleaved])

            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v_new = torch.zeros((2 * n_sel, *v.shape[1:]), device=device)
            return torch.cat([v[rest], v_new])

        _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers)

        # Assign new IDs to children (interleaved: c1_i, c2_i, c1_{i+1}, c2_{i+1}, ...)
        new_ids = torch.arange(
            self._next_id, self._next_id + 2 * n_sel, dtype=torch.long
        )
        self._next_id += 2 * n_sel

        # Record splits and store parent snapshots (atleast_1d ensures ndarray, not scalar)
        for i in range(n_sel):
            pid = parent_leaf_ids_sel[i].item()
            c1_id = new_ids[2 * i].item()
            c2_id = new_ids[2 * i + 1].item()
            parent_snap = {
                k: np.atleast_1d(parent_cpu[k][i]).astype(np.float32)
                for k in parent_cpu
            }
            self._child_to_parent[c1_id] = parent_snap
            self._child_to_parent[c2_id] = parent_snap
            self.splits.append(SplitRecord(pid, c1_id, c2_id, level))

        # New leaf_ids: [surviving | c1_0, c2_0, c1_1, c2_1, ...]
        new_leaf_ids = torch.cat([leaf_ids[rest], new_ids.to(device)])
        return new_leaf_ids

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def update_ids_after_prune(self, keep_mask: Tensor, leaf_ids: Tensor) -> Tensor:
        """Returns leaf_ids with pruned entries removed (keep_mask=True means keep)."""
        return leaf_ids[keep_mask]

    # ------------------------------------------------------------------
    # Post-training residual extraction
    # ------------------------------------------------------------------

    def extract_residuals(
        self, splats: torch.nn.ParameterDict, leaf_ids: Tensor
    ) -> List[dict]:
        """Compute (ψ, α) for each surviving split from final trained leaf params.

        Returns a list of dicts, one per split where both children survived pruning.
        Each dict has keys: parent_id, c1_id, c2_id, level, psi (np.ndarray), alpha (np.ndarray).
        """
        id_to_pos = {leaf_ids[i].item(): i for i in range(len(leaf_ids))}
        results = []

        for sr in self.splits:
            parent_np = self._child_to_parent.get(sr.child1_leaf_id)
            if parent_np is None:
                continue
            c1_pos = id_to_pos.get(sr.child1_leaf_id)
            c2_pos = id_to_pos.get(sr.child2_leaf_id)
            if c1_pos is None or c2_pos is None:
                continue  # one or both children pruned

            psi_parts: List[np.ndarray] = []
            alpha_vals: List[float] = []

            for k in _PARAM_GROUPS:
                if k not in splats:
                    psi_parts.append(np.array([], dtype=np.float32))
                    alpha_vals.append(1.0)
                    continue
                # parent_np[k] is always ≥1-D (atleast_1d stored at split time)
                parent_k = torch.from_numpy(parent_np[k]).float().flatten()
                c1_k = splats[k][c1_pos].detach().cpu().float().flatten()
                c2_k = splats[k][c2_pos].detach().cpu().float().flatten()

                psi_k = c1_k - parent_k
                diff_k = c2_k - parent_k

                # α_group: scalar asymmetry for this parameter group
                nz = psi_k.abs() > 1e-8
                if nz.any():
                    alpha_k = (-diff_k[nz] / psi_k[nz]).mean().clamp(0.01, 10.0).item()
                else:
                    alpha_k = 1.0

                psi_parts.append(psi_k.numpy())
                alpha_vals.append(alpha_k)

            results.append({
                "parent_id": sr.parent_leaf_id,
                "c1_id": sr.child1_leaf_id,
                "c2_id": sr.child2_leaf_id,
                "level": sr.level,
                "psi": np.concatenate(psi_parts).astype(np.float32),
                "alpha": np.array(alpha_vals, dtype=np.float32),
            })

        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(
        self,
        tree_dir: str,
        splats: Optional[torch.nn.ParameterDict] = None,
        leaf_ids: Optional[Tensor] = None,
    ) -> None:
        """Save topology JSON and (optionally) residuals NPZ."""
        os.makedirs(tree_dir, exist_ok=True)

        topology = {
            "n_roots": self._next_id - 2 * len(self.splits),  # approx; may be off if pruned
            "n_total_ids": self._next_id,
            "n_splits": len(self.splits),
            "splits": [
                {
                    "parent_id": s.parent_leaf_id,
                    "c1_id": s.child1_leaf_id,
                    "c2_id": s.child2_leaf_id,
                    "level": s.level,
                }
                for s in self.splits
            ],
        }
        with open(os.path.join(tree_dir, "topology.json"), "w") as f:
            json.dump(topology, f, indent=2)

        if splats is not None and leaf_ids is not None:
            residuals = self.extract_residuals(splats, leaf_ids)
            if residuals:
                psi_arr = np.stack([r["psi"] for r in residuals])
                alpha_arr = np.stack([r["alpha"] for r in residuals])
                np.savez_compressed(
                    os.path.join(tree_dir, "residuals.npz"),
                    psi=psi_arr,
                    alpha=alpha_arr,
                )
                print(
                    f"[Tree] {len(residuals)} residual records | "
                    f"psi shape {psi_arr.shape} | alpha shape {alpha_arr.shape}"
                )

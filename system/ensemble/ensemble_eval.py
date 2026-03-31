"""Ensemble evaluation utilities for FedDES."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def evaluate_ensemble(
    logits: torch.Tensor,
    ds: torch.Tensor,
    num_classes: int,
    hard_preds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate base classifier outputs using hard weighted voting.

    Each classifier's vote is weighted by ReLU(logit) — the GNN's predicted
    competence score. Classifiers with negative logits are excluded.
    Samples where all logits are negative fall back to uniform voting.

    Args:
        logits: GNN output logits [N, M].
        ds: Decision-space probabilities [N, M*C] (unused here, kept for API consistency).
        num_classes: Number of classes C.
        hard_preds: Hard predictions [N, M] from base classifiers.

    Returns:
        (soft_probs, hard_preds_out): [N, C] weighted vote distribution and [N] predictions.
    """
    weights = F.relu(logits)  # [N, M] — zero out negatives
    sum_w = weights.sum(dim=1, keepdim=True)
    fallback = (sum_w == 0)
    final_weights = torch.where(fallback, torch.ones_like(weights), weights)
    norm_weights = final_weights / final_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

    one_hot = F.one_hot(hard_preds, num_classes=num_classes).float()  # [N, M, C]
    soft_probs = (norm_weights.unsqueeze(-1) * one_hot).sum(dim=1)    # [N, C]
    return soft_probs, soft_probs.argmax(dim=1)


def compute_selection_matrix(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ReLU selection weights and fallback mask from GNN logits.

    Args:
        logits: GNN output logits [N, M].

    Returns:
        (selection_matrix [N, M], fallback_rows [N] bool).
    """
    selection_matrix = F.relu(logits)
    fallback_rows = selection_matrix.sum(dim=1) == 0
    if fallback_rows.any():
        selection_matrix = torch.where(
            fallback_rows.unsqueeze(1), torch.ones_like(selection_matrix), selection_matrix
        )
    return selection_matrix, fallback_rows


# ---------------------------------------------------------------------------
# Effective ensemble size diagnostics
# ---------------------------------------------------------------------------

def ess_stats(weights: torch.Tensor) -> dict:
    """Effective ensemble size (ESS) statistics from selection weights.

    ESS per sample = 1 / sum(w_i^2) for normalized weights.

    Returns dict with keys: mean, q25, med, q75.
    """
    if weights is None or weights.numel() == 0:
        return {"mean": 0.0, "q25": 0.0, "med": 0.0, "q75": 0.0}
    w = weights.detach().float()
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    ess = 1.0 / w.pow(2).sum(dim=1).clamp(min=1e-8)
    qs = torch.quantile(ess, torch.tensor([0.25, 0.5, 0.75], device=ess.device))
    return {
        "mean": float(ess.mean().item()),
        "q25": float(qs[0].item()),
        "med": float(qs[1].item()),
        "q75": float(qs[2].item()),
    }

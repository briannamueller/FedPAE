"""Meta-label BCE loss for FedDES GNN meta-learner training."""
from __future__ import annotations

import torch


def _balance_per_elem(
    per_elem: torch.Tensor,
    meta: torch.Tensor,
    mode: str = "full",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Reweight per-element losses to balance positive/negative meta-labels.

    Modes:
      ``"full"``: 50/50 balance — positives and negatives receive equal total weight.
      ``"sqrt"``: Moderate sqrt-ratio reweighting.

    Args:
        per_elem: Unreduced loss [B, M].
        meta: Meta-label tensor [B, M] (binary).
        mode: ``"full"`` or ``"sqrt"``.
        threshold: Boundary between positive and negative elements.

    Returns:
        Reweighted per-element tensor [B, M].
    """
    pos_mask = (meta >= threshold).float()
    neg_mask = 1.0 - pos_mask
    n_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
    n_neg = neg_mask.sum(dim=1, keepdim=True).clamp(min=1)
    M = float(per_elem.size(1))

    if mode == "full":
        w_pos = (M / 2.0) / n_pos
        w_neg = (M / 2.0) / n_neg
    else:  # sqrt
        raw_w_pos = n_neg.sqrt()
        raw_w_neg = n_pos.sqrt()
        total = (n_pos * raw_w_pos + n_neg * raw_w_neg).clamp(min=1e-8)
        w_pos = M * raw_w_pos / total
        w_neg = M * raw_w_neg / total

    return per_elem * (pos_mask * w_pos + neg_mask * w_neg)


def compute_meta_loss(
    logits: torch.Tensor,
    train_meta: torch.Tensor,
    criterion_none: torch.nn.Module,
    sample_weights: torch.Tensor | None = None,
    balance_meta_elements: str = "none",
) -> torch.Tensor:
    """Binary cross-entropy meta-label loss.

    Trains the GNN to predict which classifiers are correct on each sample
    (meta_labels_bce: BCE against the binary oracle meta-labels).

    Args:
        logits: GNN output logits [B, M].
        train_meta: Binary meta-labels [B, M] (1 = classifier correct on this sample).
        criterion_none: BCEWithLogitsLoss(reduction="none").
        sample_weights: Optional per-sample weights [B].
        balance_meta_elements: ``"none"``, ``"sqrt"``, or ``"full"`` — reweights
            per-element losses to offset positive/negative meta-label imbalance.

    Returns:
        Scalar loss.
    """
    per_elem = criterion_none(logits, train_meta)  # [B, M]

    balance = str(balance_meta_elements).lower()
    if balance in ("sqrt", "full"):
        per_elem = _balance_per_elem(per_elem, train_meta, mode=balance)

    per_sample = per_elem.mean(dim=1)  # [B]

    if sample_weights is not None:
        per_sample = per_sample * sample_weights.to(per_sample.device)

    return per_sample.mean()

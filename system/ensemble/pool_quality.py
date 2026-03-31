"""Pool quality analysis for FedDES classifier pruning.

Computes per-classifier balanced accuracy averaged across all clients'
training data (with OOF-corrected predictions for local classifiers),
and produces diagnostic visualizations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_pool_quality_scores(
    clients: List[Any],
    global_clf_keys: List[Tuple[str, str]],
    num_classes: int,
) -> Dict[str, Any]:
    """Compute per-classifier balanced accuracy averaged across all clients.

    For each client, loads the graph bundle's train split (which already has
    OOF-corrected predictions for the client's own classifiers).  Then for
    each classifier j, computes balanced accuracy on that client's training
    labels.  Finally averages across clients (unweighted, so each hospital
    counts equally — aligned with balanced accuracy as the target metric).

    Returns a dict with:
        - ``per_clf_scores``: list of dicts, one per classifier, with keys
          ``clf_key``, ``clf_index``, ``mean_bacc``, ``per_client_bacc``,
          ``mean_minority_recall``, ``per_client_minority_recall``.
        - ``summary``: high-level stats (mean, median, min, max, etc.)
    """
    M = len(global_clf_keys)
    minority_class = 1  # shock / ARF / mortality are always class 1

    # Accumulate per-classifier, per-client scores.
    # Shape conceptually: [num_clients, M] for each metric.
    all_client_baccs: List[np.ndarray] = []
    all_client_min_recalls: List[np.ndarray] = []
    all_client_accs: List[np.ndarray] = []
    all_client_aucs: List[np.ndarray] = []
    client_roles: List[str] = []

    for client in clients:
        from ensemble.graph_utils import _bundle_filename
        pool_sfx = getattr(client, "_pool_suffix", "") or getattr(client, "pool_suffix", "")
        bundle_path = Path(client.base_dir) / _bundle_filename(client.role, pool_sfx)
        if not bundle_path.exists():
            print(f"[PoolQuality][warn] Missing graph bundle for {client.role}, skipping.")
            continue

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        train_data = bundle.get("train")
        if train_data is None:
            print(f"[PoolQuality][warn] No train split in bundle for {client.role}, skipping.")
            continue

        preds = train_data["preds"]  # [N, M]
        y = train_data["y"]          # [N]
        ds = train_data.get("ds")    # [N, M*C] flattened probs (may be absent)

        if preds.dim() != 2 or preds.size(1) != M:
            print(
                f"[PoolQuality][warn] Unexpected preds shape {tuple(preds.shape)} "
                f"for {client.role} (expected [N, {M}]), skipping."
            )
            continue

        preds_np = preds.numpy().astype(np.int64)
        y_np = y.numpy().astype(np.int64)

        # Reshape decision space for AUC computation.
        probs_np = None
        if ds is not None and ds.numel() == preds.size(0) * M * num_classes:
            probs_np = ds.float().numpy().reshape(preds.size(0), M, num_classes)

        # Per-classifier balanced accuracy, minority recall, accuracy, and AUC.
        client_baccs = np.zeros(M, dtype=np.float64)
        client_min_recalls = np.zeros(M, dtype=np.float64)
        client_accs = np.zeros(M, dtype=np.float64)
        client_aucs = np.full(M, np.nan, dtype=np.float64)

        for j in range(M):
            pred_j = preds_np[:, j]
            per_class_recall = []
            for c in range(num_classes):
                mask = y_np == c
                n_c = mask.sum()
                if n_c == 0:
                    continue
                recall_c = float((pred_j[mask] == c).sum()) / n_c
                per_class_recall.append(recall_c)

                if c == minority_class:
                    client_min_recalls[j] = recall_c

            if per_class_recall:
                client_baccs[j] = float(np.mean(per_class_recall))
            else:
                client_baccs[j] = 0.0

            client_accs[j] = float((pred_j == y_np).mean())

            # AUC for classifier j.
            if probs_np is not None and len(np.unique(y_np)) >= 2:
                try:
                    if num_classes == 2:
                        client_aucs[j] = roc_auc_score(y_np, probs_np[:, j, 1])
                    else:
                        client_aucs[j] = roc_auc_score(
                            y_np, probs_np[:, j, :],
                            multi_class="ovr", average="macro",
                        )
                except (ValueError, IndexError):
                    pass

        all_client_baccs.append(client_baccs)
        all_client_min_recalls.append(client_min_recalls)
        all_client_accs.append(client_accs)
        all_client_aucs.append(client_aucs)
        client_roles.append(client.role)

    if not all_client_baccs:
        print("[PoolQuality][warn] No client data available for pool quality scoring.")
        return {"per_clf_scores": [], "summary": {}}

    # Stack: [num_clients, M]
    bacc_matrix = np.stack(all_client_baccs, axis=0)
    min_recall_matrix = np.stack(all_client_min_recalls, axis=0)
    acc_matrix = np.stack(all_client_accs, axis=0)
    auc_matrix = np.stack(all_client_aucs, axis=0)

    # Average across clients (unweighted; NaN-safe for AUC).
    mean_baccs = bacc_matrix.mean(axis=0)        # [M]
    mean_min_recalls = min_recall_matrix.mean(axis=0)  # [M]
    mean_accs = acc_matrix.mean(axis=0)          # [M]
    with np.errstate(all="ignore"):
        mean_aucs = np.nanmean(auc_matrix, axis=0)  # [M]

    per_clf_scores = []
    for j in range(M):
        per_clf_scores.append({
            "clf_key": global_clf_keys[j],
            "clf_index": j,
            "mean_bacc": float(mean_baccs[j]),
            "per_client_bacc": {
                role: float(bacc_matrix[i, j])
                for i, role in enumerate(client_roles)
            },
            "mean_minority_recall": float(mean_min_recalls[j]),
            "per_client_minority_recall": {
                role: float(min_recall_matrix[i, j])
                for i, role in enumerate(client_roles)
            },
            "mean_acc": float(mean_accs[j]),
            "per_client_acc": {
                role: float(acc_matrix[i, j])
                for i, role in enumerate(client_roles)
            },
            "mean_auc": float(mean_aucs[j]),
            "per_client_auc": {
                role: float(auc_matrix[i, j])
                for i, role in enumerate(client_roles)
            },
        })

    summary = {
        "num_classifiers": M,
        "num_clients_evaluated": len(client_roles),
        "bacc_mean": float(mean_baccs.mean()),
        "bacc_median": float(np.median(mean_baccs)),
        "bacc_min": float(mean_baccs.min()),
        "bacc_max": float(mean_baccs.max()),
        "bacc_std": float(mean_baccs.std()),
        "minority_recall_mean": float(mean_min_recalls.mean()),
        "minority_recall_median": float(np.median(mean_min_recalls)),
        "minority_recall_min": float(mean_min_recalls.min()),
        "minority_recall_max": float(mean_min_recalls.max()),
    }

    return {
        "per_clf_scores": per_clf_scores,
        "summary": summary,
        "bacc_matrix": bacc_matrix,
        "min_recall_matrix": min_recall_matrix,
        "client_roles": client_roles,
        "global_clf_keys": global_clf_keys,
    }


def save_pool_quality_report(
    scores: Dict[str, Any],
    out_dir: Path,
) -> Path:
    """Save a JSON report of per-classifier quality scores.

    Returns the path to the saved report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "pool_quality_report.json"

    # Build a JSON-serializable version (no numpy arrays).
    serializable = {
        "summary": scores.get("summary", {}),
        "per_classifier": [],
    }
    for entry in scores.get("per_clf_scores", []):
        serializable["per_classifier"].append({
            "clf_key": [str(k) for k in entry["clf_key"]],
            "clf_index": entry["clf_index"],
            "mean_bacc": entry["mean_bacc"],
            "mean_minority_recall": entry["mean_minority_recall"],
            "per_client_bacc": entry["per_client_bacc"],
            "per_client_minority_recall": entry["per_client_minority_recall"],
        })

    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[PoolQuality] Saved report to {report_path}")
    return report_path


def save_pool_quality_plots(
    scores: Dict[str, Any],
    out_dir: Path,
) -> Path:
    """Save diagnostic visualizations of classifier pool quality.

    Produces a multi-panel figure:
      - Top-left: histogram of mean balanced accuracy across classifiers
      - Top-right: histogram of mean minority recall across classifiers
      - Bottom-left: sorted bar chart of mean balanced accuracy per classifier
        with a horizontal line at 0.5 (random baseline)
      - Bottom-right: heatmap of per-classifier, per-client balanced accuracy

    Returns the path to the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "pool_quality_diagnostics.png"

    per_clf = scores.get("per_clf_scores", [])
    if not per_clf:
        print("[PoolQuality][warn] No classifier scores to plot.")
        return plot_path

    mean_baccs = np.array([e["mean_bacc"] for e in per_clf])
    mean_min_recalls = np.array([e["mean_minority_recall"] for e in per_clf])
    clf_labels = [
        f"{e['clf_key'][0]}\n{e['clf_key'][1]}" for e in per_clf
    ]
    M = len(per_clf)

    # Sort by mean_bacc for the bar chart.
    sort_idx = np.argsort(mean_baccs)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Classifier Pool Quality Diagnostics", fontsize=16, fontweight="bold")

    # --- Top-left: Balanced accuracy histogram ---
    ax = axes[0, 0]
    n_bins = min(30, max(10, M // 2))
    ax.hist(mean_baccs, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, label="Random (0.5)")
    ax.axvline(x=np.median(mean_baccs), color="orange", linestyle="-.", linewidth=2,
               label=f"Median ({np.median(mean_baccs):.3f})")
    ax.set_xlabel("Mean Balanced Accuracy (across clients)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Cross-Client Mean Balanced Accuracy", fontsize=13)
    ax.legend(fontsize=10)

    # --- Top-right: Minority recall histogram ---
    ax = axes[0, 1]
    ax.hist(mean_min_recalls, bins=n_bins, color="coral", edgecolor="black", alpha=0.8)
    ax.axvline(x=np.median(mean_min_recalls), color="orange", linestyle="-.", linewidth=2,
               label=f"Median ({np.median(mean_min_recalls):.3f})")
    ax.set_xlabel("Mean Minority Recall (across clients)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Cross-Client Mean Minority Recall", fontsize=13)
    ax.legend(fontsize=10)

    # --- Bottom-left: Sorted bar chart of mean balanced accuracy ---
    ax = axes[1, 0]
    x_pos = np.arange(M)
    sorted_baccs = mean_baccs[sort_idx]
    sorted_min_recalls = mean_min_recalls[sort_idx]
    sorted_labels = [clf_labels[i] for i in sort_idx]

    # Color bars by whether they're above or below 0.5.
    colors = ["steelblue" if b >= 0.5 else "salmon" for b in sorted_baccs]
    bars = ax.bar(x_pos, sorted_baccs, color=colors, edgecolor="black", alpha=0.85, width=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2, label="Random baseline (0.5)")
    ax.set_xlabel("Classifier (sorted by balanced accuracy)", fontsize=12)
    ax.set_ylabel("Mean Balanced Accuracy", fontsize=12)
    ax.set_title("Per-Classifier Mean Balanced Accuracy (sorted)", fontsize=13)
    ax.set_ylim(0, max(1.0, sorted_baccs.max() + 0.05))
    ax.legend(fontsize=10)

    # Only show tick labels if the pool is small enough to be readable.
    if M <= 30:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
        # Annotate a few key classifiers.
        for pos in [0, M // 4, M // 2, 3 * M // 4, M - 1]:
            if pos < M:
                ax.annotate(
                    sorted_labels[pos],
                    (x_pos[pos], sorted_baccs[pos]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    fontsize=6,
                    ha="center",
                    rotation=45,
                )

    # --- Bottom-right: Heatmap of per-classifier balanced accuracy across clients ---
    ax = axes[1, 1]
    bacc_matrix = scores.get("bacc_matrix")
    client_roles = scores.get("client_roles", [])

    if bacc_matrix is not None and bacc_matrix.size > 0:
        # Sort columns (classifiers) same as bar chart; rows (clients) by role name.
        heatmap_data = bacc_matrix[:, sort_idx]
        im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.8)
        ax.set_xlabel("Classifier (sorted by mean bacc)", fontsize=12)
        ax.set_ylabel("Client", fontsize=12)
        ax.set_title("Per-Client Balanced Accuracy Heatmap", fontsize=13)
        fig.colorbar(im, ax=ax, label="Balanced Accuracy", shrink=0.8)

        if len(client_roles) <= 30:
            ax.set_yticks(range(len(client_roles)))
            ax.set_yticklabels(client_roles, fontsize=7)
        else:
            ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center", fontsize=14)
        ax.set_title("Per-Client Balanced Accuracy Heatmap", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[PoolQuality] Saved diagnostics plot to {plot_path}")
    return plot_path

import csv
import hashlib
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.models import BaseHeadSplit

try:
    import wandb
except Exception:
    wandb = None

_VALID_ES_METRICS = {"val_loss", "val_temp_scaled_loss"}


# ---------------------------------------------------------------------------
# Deterministic per-model seeding
# ---------------------------------------------------------------------------

def seed_for_model(
    client_id: int, model_id: int, fold_idx: int = 0,
    *, hospital_id: int = None,
) -> None:
    """Set deterministic seed based on (client_id, model_id, fold_idx).

    Ensures training results are independent of which other models are
    trained in the same run, so cached checkpoints are valid regardless
    of training order or ``base_single_model`` setting.

    When *hospital_id* is provided (pool mode), it is used instead of
    *client_id* so that the seed is stable across different filter
    selections that assign different positional indices.
    """
    key_id = hospital_id if hospital_id is not None else client_id
    seed = hash((key_id, model_id, fold_idx)) % (2**31)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pool composition hash
# ---------------------------------------------------------------------------

def pool_composition_hash(
    global_clf_keys: Sequence[Tuple[str, str]],
) -> str:
    """Return a 4-char hex hash identifying the classifier pool composition.

    Used as a suffix on bundle/feats filenames so that different pool
    compositions (e.g. base_single_model True vs False, or different
    pruning outcomes) don't collide in the same ``base_dir``.
    """
    blob = json.dumps(
        sorted([list(k) for k in global_clf_keys]), sort_keys=True
    ).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:4]

# Lazy singleton — only initialized when val_temp_scaled_loss is actually used.
_metrics_singleton = None

def _get_metrics():
    """Return the probmetrics Metrics object, creating it on first call."""
    global _metrics_singleton
    if _metrics_singleton is None:
        from probmetrics.metrics import Metrics
        _metrics_singleton = Metrics.from_names(['refinement_logloss_ts-mix_all'])
    return _metrics_singleton


_TREE_MODEL_PREFIXES = ("XGBoost", "RandomForest")


def is_tree_model(model_str: str) -> bool:
    """Return True if *model_str* refers to a tree-based (non-PyTorch) classifier."""
    return any(model_str.startswith(p) for p in _TREE_MODEL_PREFIXES)


def aggregate_timeseries(x: np.ndarray) -> np.ndarray:
    """Aggregate a time-series array into a fixed-length feature vector.

    Args:
        x: Array of shape ``[T, F]`` (single sample) or ``[N, T, F]`` (batch).

    Returns:
        Array of shape ``[F*5]`` or ``[N, F*5]`` using per-feature
        mean, std, min, max, and last-timestep value.
    """
    if x.ndim == 2:
        # Single sample [T, F]
        return np.concatenate([
            x.mean(axis=0),
            x.std(axis=0),
            x.min(axis=0),
            x.max(axis=0),
            x[-1],
        ]).astype(np.float32)
    # Batch [N, T, F]
    return np.concatenate([
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        x[:, -1, :],
    ], axis=1).astype(np.float32)


def _extract_tree_features_from_ts_static(
    ts_np: np.ndarray, static_np: np.ndarray,
) -> np.ndarray:
    """Build tree-model features from separate temporal and static arrays.

    For each sample: last timestep values (first 87 channels) + summary
    stats (mean, min, max of first 87 channels) + full static vector.

    Args:
        ts_np: ``[N, T, F_ts]`` temporal features (87 values + 87 masks).
        static_np: ``[N, F_static]`` static features (65 flat + 316 diag).

    Returns:
        ``[N, 87 + 87*3 + F_static]`` feature matrix.
    """
    n_value_channels = ts_np.shape[2] // 2  # 87 values, 87 masks
    values = ts_np[:, :, :n_value_channels]  # [N, T, 87]
    last = values[:, -1, :]                  # [N, 87]
    mean = values.mean(axis=1)               # [N, 87]
    vmin = values.min(axis=1)                # [N, 87]
    vmax = values.max(axis=1)                # [N, 87]
    return np.concatenate([last, mean, vmin, vmax, static_np], axis=1).astype(np.float32)


def _extract_numpy_from_loader(loader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract all (x, y) from a DataLoader as numpy arrays.

    Handles both single-tensor and ``(temporal, static)`` tuple formats.

    Returns:
        x_all: flat feature array for tree models.
        y_all: ``[N]`` integer labels.
    """
    xs, ys = [], []
    for batch in loader:
        x, y = batch
        y = y[0] if isinstance(y, (list, tuple)) else y
        y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

        # Two-input format: (temporal, static)
        if isinstance(x, (list, tuple)) and len(x) == 2 and all(torch.is_tensor(t) for t in x):
            ts_np = x[0].cpu().numpy()
            static_np = x[1].cpu().numpy()
            x_np = _extract_tree_features_from_ts_static(ts_np, static_np)
        else:
            x = x[0] if isinstance(x, (list, tuple)) else x
            x_np = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
            if x_np.ndim == 3:
                x_np = aggregate_timeseries(x_np)
            elif x_np.ndim == 2:
                pass  # already flat
            else:
                raise ValueError(f"Unexpected x shape: {x_np.shape}")
        xs.append(x_np)
        ys.append(y_np.astype(np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_tree_clf(
    client,
    model_id: int,
    model_str: str,
    train_loader,
    val_loader,
    **kwargs,
):
    """Train an XGBoost or RandomForest classifier.

    Returns the same ``(best_iteration, best_score, model)`` triple as
    :func:`fit_clf` so the caller can treat neural and tree models uniformly.
    """
    x_train, y_train = _extract_numpy_from_loader(train_loader)
    x_val, y_val = _extract_numpy_from_loader(val_loader)

    num_classes = int(getattr(client.args, "num_classes", 2))

    if model_str.startswith("XGBoost"):
        from xgboost import XGBClassifier
        # Class imbalance: scale_pos_weight for binary, sample_weight for multi
        xgb_params = dict(
            n_estimators=int(getattr(client.args, "xgb_n_estimators", 200)),
            max_depth=int(getattr(client.args, "xgb_max_depth", 5)),
            learning_rate=float(getattr(client.args, "xgb_learning_rate", 0.1)),
            subsample=float(getattr(client.args, "xgb_subsample", 0.8)),
            colsample_bytree=float(getattr(client.args, "xgb_colsample_bytree", 0.8)),
            min_child_weight=int(getattr(client.args, "xgb_min_child_weight", 1)),
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=hash((client.id, model_id)) % (2**31),
        )
        if num_classes == 2:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            xgb_params["scale_pos_weight"] = max(0.1, min(50.0, n_neg / max(n_pos, 1)))
        model = XGBClassifier(**xgb_params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        best_iteration = int(getattr(model, "best_iteration", xgb_params["n_estimators"]))
        y_val_pred = model.predict(x_val)
        best_score = float((y_val_pred == y_val).mean())

    elif model_str.startswith("RandomForest"):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=int(getattr(client.args, "rf_n_estimators", 500)),
            max_depth=int(getattr(client.args, "rf_max_depth", 10)),
            class_weight="balanced",
            random_state=hash((client.id, model_id)) % (2**31),
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        best_score = float((y_val_pred == y_val).mean())
        best_iteration = 0
    else:
        raise ValueError(f"Unknown tree model: {model_str}")

    print(f"[TreeClf] {client.role} {model_str} val_acc={best_score:.4f}")
    return best_iteration, best_score, model


def process_batch(batch, device: torch.device):
    """
    Normalize a batch to (x, y) tensors on the target device.

    ``x`` may be a single tensor or a ``(temporal, static)`` tuple for
    TPC two-input models.  In the tuple case both tensors are moved to
    *device* and the tuple structure is preserved.
    """
    x, y = batch
    y = y[0] if isinstance(y, (list, tuple)) else y

    if isinstance(x, (list, tuple)):
        # Two-tensor input: (temporal, static)
        if len(x) == 2 and all(torch.is_tensor(t) for t in x):
            x = (x[0].to(device), x[1].to(device))
        else:
            # Legacy single-tensor wrapped in list/tuple
            x = x[0] if isinstance(x, (list, tuple)) else x
            x = x.to(device)
    else:
        x = x.to(device)
    return x, y.to(device)


# -------------------------------
# Core epoch helpers
# -------------------------------

def train_one_epoch(model, loader, device, optimizer, loss_fn, scheduler=None):
    """Single training epoch: cross-entropy loss + accuracy."""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for batch in loader:
        x, y = process_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return {"loss": avg_loss, "acc": acc}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, return_logits=False):
    """
    Evaluate loss + accuracy.

    If return_logits=True, also return (logits_cat, labels_cat) for temp scaling.
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_logits = [] if return_logits else None
    all_labels = [] if return_logits else None

    for batch in loader:
        x, y = process_batch(batch, device)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total += y.size(0)

        if return_logits:
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    stats = {"loss": avg_loss, "acc": acc}

    if return_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        return stats, logits_cat, labels_cat

    return stats


# -------------------------------
# Loss builder
# -------------------------------

class BalancedBCEWithLogits(nn.Module):
    """Binary cross-entropy with logits using a positive-class weight."""
    def __init__(self, pos_weight: float):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 1 and logits.size(1) == 2:
            logits = logits[:, 1]           # keep only the logit for class 1
        targets = targets.float()
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
        )


def build_loss_fn(
    num_classes: int,
    weighted: bool,
    device: torch.device,
    train_dataset=None,
):
    """Binary: BCEWithLogits (optional pos_weight); Multi-class: CE (optional weights)."""

    # Binary case
    if num_classes == 2:
        if not weighted or train_dataset is None:
            return BalancedBCEWithLogits(pos_weight=1.0)

        counts = torch.zeros(num_classes, dtype=torch.float)
        for _, y in train_dataset:
            lbl = int(y.item()) if torch.is_tensor(y) else int(y)
            if 0 <= lbl < counts.numel():
                counts[lbl] += 1.0
        pos_weight = (counts[0] / counts[1].clamp(min=1.0)).clamp(max=50.0)
        return BalancedBCEWithLogits(pos_weight=pos_weight.to(device))

    # Multi-class case
    if not weighted or train_dataset is None:
        return nn.CrossEntropyLoss()

    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, y in train_dataset:
        lbl = int(y.item()) if torch.is_tensor(y) else int(y)
        if 0 <= lbl < counts.numel():
            counts[lbl] += 1.0

    if counts.sum() == 0:
        return nn.CrossEntropyLoss()

    safe = counts.clamp(min=1.0)
    weights = (safe.sum() / safe.numel()) / safe
    return nn.CrossEntropyLoss(weight=weights.to(device))


# -------------------------------
# W&B logging helper
# -------------------------------

def _log_wandb_run(
    client,
    model_id,
    history: list,
    fieldnames: list,
    es_metric: str,
    best_metric: float,
    best_epoch: int,
    log_wandb: Optional[bool],
):
    """Log base classifier training history to W&B.  No-op if W&B is disabled."""
    if wandb is None or log_wandb is False:
        return
    flag = os.getenv("WANDB_BASE_LOG", "")
    if flag and flag.lower() in {"0", "false"}:
        return
    if not history or not fieldnames:
        return

    # --- init run ---
    run_config = {
        "batch_size": getattr(client.args, "batch_size", None),
        "feature_dim": getattr(client.args, "feature_dim", None),
        "dataset": getattr(client.args, "data_partition", None),
    }
    for key, value in vars(client.args).items():
        if key.startswith("base_"):
            run_config[key] = value

    try:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"{client.role}_model_{model_id}",
            reinit=True,
            config=run_config,
            job_type="base_clf",
        )
    except Exception:
        return
    if run is None:
        return

    # --- build tables & plots ---
    log_payload = {}
    try:
        table_data = [[row.get(col) for col in fieldnames] for row in history]
        history_table = wandb.Table(columns=fieldnames, data=table_data)

        log_payload[f"base_model_{model_id}_history"] = history_table
        log_payload[f"base_model_{model_id}_train_loss"] = wandb.plot.line(
            history_table, "epoch", "train_loss",
            title=f"Base model {model_id} train loss",
        )
        log_payload[f"base_model_{model_id}_val_loss"] = wandb.plot.line(
            history_table, "epoch", "val_loss",
            title=f"Base model {model_id} val loss",
        )
        if es_metric != "val_loss" and "es_metric" in fieldnames:
            log_payload[f"base_model_{model_id}_es_metric"] = wandb.plot.line(
                history_table, "epoch", "es_metric",
                title=f"Base model {model_id} {es_metric}",
            )
        run.log(log_payload)
    except Exception:
        pass

    # --- summary cleanup ---
    try:
        if hasattr(run, "summary") and run.summary is not None:
            for key in log_payload:
                try:
                    del run.summary[key]
                except Exception:
                    pass
            es_label = "val_ts_ref_loss" if es_metric == "val_temp_scaled_loss" else "val_loss"
            run.summary.update({
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "early_stopping_metric": es_label,
            })
    except Exception:
        pass

    try:
        wandb.finish()
    except Exception:
        pass


# -------------------------------
# Main training function
# -------------------------------

def fit_clf(
    client,
    model_id,
    train_loader,
    val_loader,
    device,
    max_epochs=100,
    patience=10,
    min_delta=0.0,
    es_metric="val_loss",
    lr=1e-3,
    warmup_epochs=10,
    log_wandb: Optional[bool] = None,
):
    if es_metric not in _VALID_ES_METRICS:
        raise ValueError(
            f"Unknown es_metric={es_metric!r}. Valid options: {sorted(_VALID_ES_METRICS)}"
        )

    model = BaseHeadSplit(client.args, model_id).to(device)

    loss_fn = build_loss_fn(
        client.args.num_classes,
        getattr(client.args, "base_weighted_by_class", True),
        device,
        train_dataset=train_loader.dataset,
    )
    # Unweighted loss for validation — ensures val_loss is a clean generalization
    # signal regardless of whether training uses class weights.
    eval_loss_fn = build_loss_fn(client.args.num_classes, False, device)

    opt_name = getattr(client.args, "base_optimizer", "Adam").upper()
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    warmup_scheduler = None
    if warmup_epochs > 0 and len(train_loader) > 0:
        warmup_steps = warmup_epochs * len(train_loader)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / warmup_steps),
        )

    best_state = None
    best_metric = -float("inf")
    best_epoch, stale_epochs = 0, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        train_stats = train_one_epoch(
            model, train_loader, device, optimizer, loss_fn,
            scheduler=warmup_scheduler,
        )

        # ---- validate ----
        if es_metric == "val_temp_scaled_loss":
            val_stats, val_logits, val_labels = evaluate(
                model, val_loader, device, eval_loss_fn, return_logits=True,
            )
            ts_results = _get_metrics().compute_all_from_labels_logits(val_labels, val_logits)
            val_stats["ts_ref_loss"] = ts_results['refinement_logloss_ts-mix_all'].item()
            metric_val = -val_stats["ts_ref_loss"]
        else:
            val_stats = evaluate(model, val_loader, device, eval_loss_fn)
            metric_val = -val_stats["loss"]

        # ---- log row ----
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
        }
        if es_metric == "val_temp_scaled_loss":
            row["val_ts_ref_loss"] = val_stats["ts_ref_loss"]
            row["es_metric"] = val_stats["ts_ref_loss"]
        else:
            row["es_metric"] = val_stats["loss"]
        history.append(row)

        # ---- checkpoint ----
        improved = metric_val > best_metric + float(min_delta)
        if improved:
            best_metric = metric_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"[BaseClf][EarlyStop] epoch={epoch}, best_metric={best_metric:.6f}")
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------------------
    # Save training logs to CSV
    # ---------------------------
    logs_dir = client.base_outputs_dir / f"{client.role}_training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"model_{model_id}.csv"

    fieldnames: List[str] = []
    if history:
        fieldnames = sorted({k for row in history for k in row.keys()})
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

    print(f"[BaseClf] Saved training logs to {log_path}")

    # ---------------------------
    # Log to W&B (extracted helper)
    # ---------------------------
    _log_wandb_run(
        client, model_id, history, fieldnames,
        es_metric, best_metric, best_epoch, log_wandb,
    )

    return best_epoch, best_metric, model

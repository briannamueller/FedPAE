
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models
from ensemble.edge_builders import build_cs_edges, build_ss_edges_cmdw
from ensemble.base_clf_utils import process_batch, is_tree_model, aggregate_timeseries
from ensemble.dataset_stats import load_client_label_counts
from torch_geometric.data import HeteroData
from probmetrics.calibrators import get_calibrator
from probmetrics.distributions import CategoricalLogits


def _resolve_attr(module: torch.nn.Module, attr: str):
    if hasattr(module, attr):
        return getattr(module, attr)
    if hasattr(module, "module") and hasattr(module.module, attr):
        return getattr(module.module, attr)
    return None


def _forward_with_embedding(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    base = _resolve_attr(model, "base")
    head = _resolve_attr(model, "head")
    if base is not None and head is not None:
        rep = base(x)
        if isinstance(rep, tuple):
            rep = rep[0]
        rep = rep.contiguous()
        logits = head(rep)
        rep_flat = rep.view(rep.size(0), -1)
        return logits, rep_flat

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.contiguous()
    rep_flat = logits.view(logits.size(0), -1)
    return logits, rep_flat


def compute_meta_labels(probs, preds, labels, min_positive=5):
    mask = (preds == labels.unsqueeze(1)).clone()

    with torch.no_grad():
        true_cls_probs = probs[torch.arange(probs.size(0)), :, labels]  # [N, M]
        for i in range(probs.size(0)):
            needed = min_positive - int(mask[i].sum().item())
            if needed <= 0:
                continue
            for idx in torch.argsort(true_cls_probs[i], descending=True):
                if mask[i, idx]:
                    continue
                mask[i, idx] = True
                needed -= 1
                if needed <= 0:
                    break
    return mask.to(torch.uint8)


def compute_auc_meta_labels(
    ds: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    max_pairs: int = 5000,
) -> torch.Tensor:
    """Per-sample AUC contribution meta-labels [N, M].

    For each class c (one-vs-rest), computes what fraction of opposite-class
    samples each classifier ranks correctly relative to each sample.  Results
    are macro-averaged across classes.

    Args:
        ds: Decision-space tensor [N, M*C] (flattened probabilities).
        y:  True labels [N].
        num_classes: Number of classes C.
        max_pairs: Cap on opposite-class samples to compare against (per class)
            to keep computation tractable.  If n_neg > max_pairs, a random
            subsample is used.

    Returns:
        Tensor [N, M] with values in [0, 1].
    """
    M = ds.size(1) // num_classes
    probs = ds.view(-1, M, num_classes).float()  # [N, M, C]
    N = probs.size(0)
    auc_meta = torch.zeros(N, M)

    gen = torch.Generator()
    gen.manual_seed(0)

    classes_present = y.unique()
    n_classes_present = len(classes_present)
    if n_classes_present < 2:
        return auc_meta

    for c in classes_present.tolist():
        pos_mask = (y == c)
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos == 0 or n_neg == 0:
            continue

        # P(class=c) for each sample under each classifier
        p_c = probs[:, :, c]  # [N, M]
        p_pos = p_c[pos_mask]  # [n_pos, M]
        p_neg = p_c[neg_mask]  # [n_neg, M]

        # Subsample negatives if too many
        if n_neg > max_pairs:
            idx = torch.randperm(n_neg, generator=gen)[:max_pairs]
            p_neg = p_neg[idx]
            n_neg = max_pairs

        # For positive samples: fraction of negatives ranked below
        # p_pos[:, None, :] vs p_neg[None, :, :] → [n_pos, n_neg, M]
        ranks_pos = (p_pos.unsqueeze(1) > p_neg.unsqueeze(0)).float().mean(dim=1)  # [n_pos, M]

        # Subsample positives if too many for neg computation
        p_pos_for_neg = p_pos
        if n_pos > max_pairs:
            idx = torch.randperm(n_pos, generator=gen)[:max_pairs]
            p_pos_for_neg = p_pos[idx]

        # For negative samples: fraction of positives ranked above
        ranks_neg = (p_pos_for_neg.unsqueeze(1) > p_neg.unsqueeze(0)).float().mean(dim=0)  # [n_neg, M]

        auc_meta[pos_mask] += ranks_pos / n_classes_present
        auc_meta[neg_mask] += ranks_neg / n_classes_present

    return auc_meta


def project_to_DS(self, loader, classifier_pool, calibrate_probs=True, feat_mode_override=None):

    classifier_keys = self.global_clf_keys
    num_classifiers = len(classifier_keys)

    logits_all = []
    embeddings_all = []
    batches = list(loader)

    labels_all = []
    for batch in batches:
        _, y = batch
        y = y[0] if isinstance(y, (list, tuple)) else y
        labels_all.append(y.detach().cpu())

    # Identify which classifier keys are tree-based vs neural
    # key[1] is a checkpoint name like "model_2"; resolve the actual model def.
    def _is_tree_key(key):
        if not isinstance(key, (list, tuple)):
            return False
        model_id = int(key[1].split("_")[-1])
        return is_tree_model(self.args.models[model_id])
    tree_flags = [_is_tree_key(key) for key in classifier_keys]
    has_trees = any(tree_flags)

    # Pre-extract aggregated numpy features for tree models (done once)
    if has_trees:
        from ensemble.base_clf_utils import _extract_tree_features_from_ts_static
        _tree_x_parts = []
        for batch in batches:
            x, _ = process_batch(batch, self.device)
            if isinstance(x, tuple):
                ts_np = x[0].cpu().numpy()
                static_np = x[1].cpu().numpy()
                x_np = _extract_tree_features_from_ts_static(ts_np, static_np)
            else:
                x_np = x.cpu().numpy()
                if x_np.ndim == 3:
                    x_np = aggregate_timeseries(x_np)
            _tree_x_parts.append(x_np)
        _tree_x_all = np.concatenate(_tree_x_parts, axis=0)

    with torch.no_grad():
        for key_idx, key in enumerate(classifier_keys):
            model = classifier_pool[key]

            if tree_flags[key_idx]:
                # Tree model: predict_proba → probabilities directly
                probs_np = model.predict_proba(_tree_x_all).astype(np.float32)
                num_classes = int(getattr(self.args, "num_classes", probs_np.shape[1]))
                # Ensure probs_np has correct number of columns
                if probs_np.shape[1] < num_classes:
                    padded = np.zeros((_tree_x_all.shape[0], num_classes), dtype=np.float32)
                    padded[:, :probs_np.shape[1]] = probs_np
                    probs_np = padded
                # Use log(probs) as pseudo-logits for calibration compatibility
                eps = 1e-7
                pseudo_logits = torch.from_numpy(np.log(np.clip(probs_np, eps, 1.0)))
                logits_all.append(pseudo_logits)
                # Use aggregated features as "embedding" for tree models
                embeddings_all.append(torch.from_numpy(_tree_x_all))
            else:
                model = model.to(self.device).eval()
                per_model_logits = []
                per_model_embeds = []
                for batch in batches:
                    x, y = process_batch(batch, self.device)
                    logits, embeds = _forward_with_embedding(model, x)
                    per_model_embeds.append(embeds.cpu())
                    per_model_logits.append(logits.cpu())
                model.to("cpu").eval()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logits_all.append(torch.cat(per_model_logits, dim=0))
                embeddings_all.append(torch.cat(per_model_embeds, dim=0))

    # Stack all models: [N, M, C]
    logits = torch.stack(logits_all, dim=1)
    labels = torch.cat(labels_all, dim=0)   # [N]

    # Pad embeddings to common dimension before stacking
    max_embed_dim = max(e.size(1) for e in embeddings_all)
    padded_embeds = []
    for e in embeddings_all:
        if e.size(1) < max_embed_dim:
            pad = torch.zeros(e.size(0), max_embed_dim - e.size(1))
            e = torch.cat([e, pad], dim=1)
        padded_embeds.append(e)
    embeddings = torch.stack(padded_embeds, dim=1)

    if calibrate_probs:
        probs = _calibrate_logits(self, logits, labels, num_classifiers)
    else:
        probs = torch.softmax(logits, dim=2)

    preds = probs.argmax(dim=2)                 # [N, M]
    meta_labels = compute_meta_labels(
        probs, preds, labels,
        min_positive=int(getattr(self.args, "graph_meta_min_pos", 3)),
    )

    probs_flat = probs.reshape(probs.size(0), -1)

    feat_mode = feat_mode_override or str(getattr(self.args, "graph_sample_node_feats", "ds")).lower()

    if feat_mode == "ds":
        features = probs_flat
    elif feat_mode == "embedding_mean":
        features = embeddings.mean(dim=1)
    elif feat_mode == "embedding_concat":
        features = embeddings.reshape(embeddings.size(0), -1)
    elif feat_mode == "encoder":
        dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
        if "eicu" in dataset_name:
            features = encode_eicu_features(self, batches)
        else:
            features = encode_with_graph_encoder(self, batches)
    elif feat_mode in ("hybrid", "ds_static"):
        dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
        if "eicu" in dataset_name:
            static_feats = _extract_static_features(self, batches)
            features = torch.cat([probs_flat, static_feats], dim=1)
        else:
            features = probs_flat
    elif feat_mode == "static":
        dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
        if "eicu" in dataset_name:
            features = _extract_static_features(self, batches)
        else:
            features = probs_flat
    elif feat_mode == "demographics":
        dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
        if "eicu" in dataset_name:
            features = _extract_demographic_features(self, batches)
        else:
            features = probs_flat
    elif feat_mode == "hybrid_demographics":
        dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
        if "eicu" in dataset_name:
            demo_feats = _extract_demographic_features(self, batches)
            features = torch.cat([probs_flat, demo_feats], dim=1)
        else:
            features = probs_flat
    else:
        features = probs_flat

    return probs_flat, preds, labels, meta_labels, features


# ---------------------------------------------------------------------------
# Separate feats cache (per feature-mode)
# ---------------------------------------------------------------------------

from pathlib import Path as _Path


def _feats_filename(role: str, feat_mode: str, pool_suffix: str = "") -> str:
    """Build the feats cache filename, with optional pool suffix."""
    if pool_suffix:
        return f"{role}_feats_{feat_mode}_pool[{pool_suffix}].pt"
    return f"{role}_feats_{feat_mode}.pt"


def _bundle_filename(role: str, pool_suffix: str = "") -> str:
    """Build the graph bundle filename, with optional pool suffix."""
    if pool_suffix:
        return f"{role}_graph_bundle_pool[{pool_suffix}].pt"
    return f"{role}_graph_bundle.pt"


def load_feats(directory, role: str, feat_mode: str, pool_suffix: str = ""):
    """Load feats from a separate per-mode cache file.

    Returns:
        ``dict`` with split keys (``"train"``, ``"val"``, ``"test"``) mapping
        to feature tensors, or ``None`` if the file does not exist.
    """
    feats_path = _Path(directory) / _feats_filename(role, feat_mode, pool_suffix)
    if feats_path.exists():
        return torch.load(feats_path, map_location="cpu", weights_only=False)
    return None


def save_feats(directory, role: str, feat_mode: str, feats_data: dict, pool_suffix: str = ""):
    """Save feats to a separate per-mode cache file."""
    feats_path = _Path(directory) / _feats_filename(role, feat_mode, pool_suffix)
    feats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats_data, feats_path)


def require_feats(directory, role: str, feat_mode: str, pool_suffix: str = ""):
    """Load feats or raise a clear error if the file is missing."""
    data = load_feats(directory, role, feat_mode, pool_suffix)
    if data is None:
        fname = _feats_filename(role, feat_mode, pool_suffix)
        raise FileNotFoundError(
            f"Feats file not found: {_Path(directory) / fname}\n"
            f"Delete your cached bundles in base_dir and rerun to regenerate."
        )
    return data


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _calibrate_logits(self, logits, labels, num_classifiers):
    """Fit calibrators (once) and apply to produce calibrated probabilities."""
    calib_method = getattr(self.args, "pool_calib_method", "ts-mix")

    # Fit calibrators once and cache
    pool_calibrators = getattr(self, "_pool_calibrators", None)
    if pool_calibrators is not None and len(pool_calibrators) != num_classifiers:
        pool_calibrators = None
        setattr(self, "_pool_calibrators", None)

    if pool_calibrators is None:
        pool_calibrators = []
        for i in range(num_classifiers):
            logits_slice = logits[:, i, :]  # [N, C]
            calib = get_calibrator(calib_method)
            dist_fit = CategoricalLogits(logits_slice)
            calib.fit_torch(dist_fit, labels)
            pool_calibrators.append(calib)
        self._pool_calibrators = pool_calibrators

    # Apply calibrators
    probs_per_model = []
    for i in range(num_classifiers):
        logits_slice = logits[:, i, :]
        dist_apply = CategoricalLogits(logits_slice)
        result = self._pool_calibrators[i].predict_proba_torch(dist_apply)
        probs_i = result.get_probs()     # [N, C]
        probs_per_model.append(probs_i.unsqueeze(1))  # [N,1,C]

    return torch.cat(probs_per_model, dim=1)  # [N,M,C]


# ---------------------------------------------------------------------------
# Encoder helpers (for "encoder" / "hybrid" feature modes)
# ---------------------------------------------------------------------------

def encode_with_graph_encoder(self, batches):
    encoder = getattr(self, "_graph_encoder", None)
    if encoder is None:
        encoder = init_graph_encoder()
        self._graph_encoder = encoder.to(self.device)
    encoder.eval()

    encoded_feats = []
    with torch.no_grad():
        for batch in batches:
            x, _ = process_batch(batch, self.device)
            x = preprocess_for_encoder(x)
            feat = encoder(x)
            feat = feat.reshape(feat.size(0), -1)
            encoded_feats.append(feat.cpu())

    if not encoded_feats:
        return torch.empty(0, 0)
    return torch.cat(encoded_feats, dim=0)


def encode_eicu_features(self, batches):
    encoded_feats = []
    with torch.no_grad():
        for batch in batches:
            x, _ = process_batch(batch, self.device)
            if isinstance(x, tuple):
                # TPC two-input: mean-pool temporal, concat with static
                ts, static = x
                ts_feat = ts.mean(dim=1) if ts.dim() == 3 else ts
                feat = torch.cat([ts_feat, static], dim=1)
            elif x.dim() == 3:
                feat = x.mean(dim=1)
            else:
                feat = x.reshape(x.size(0), -1)
            encoded_feats.append(feat.cpu())

    if not encoded_feats:
        return torch.empty(0, 0)
    return torch.cat(encoded_feats, dim=0)


def _get_n_static_cols(self):
    """Return the number of static (S) columns in fused eICU data.

    Reads the partition's ``config.json`` to get ``data_dir``, ``task``,
    ``theta_1``, ``theta_2``, then loads ``S.npz`` from the FIDDLE output
    directory to determine the column count.  Result is cached on ``self``.
    """
    cached = getattr(self, "_n_static_cols", None)
    if cached is not None:
        return cached

    import json as _json
    from pathlib import Path

    dataset = getattr(self, "dataset", None) or getattr(self.args, "dataset", "")
    cfg_path = Path("..") / "dataset" / dataset / "config.json"
    # Walk up to find config.json (pool-mode selection subdirs may not have one)
    if not cfg_path.exists():
        for parent in cfg_path.parents:
            candidate = parent / "config.json"
            if candidate.exists():
                cfg_path = candidate
                break
    with open(cfg_path) as f:
        cfg = _json.load(f)

    # If generate_eICU already wrote n_static_cols, use it directly.
    if "n_static_cols" in cfg:
        self._n_static_cols = int(cfg["n_static_cols"])
        print(f"[static_feats] n_static_cols={self._n_static_cols} (from config.json n_static_cols)")
        return self._n_static_cols

    # Try n_static_features (written by both generate_eICU.py and generate_eICU_tpc.py).
    if "n_static_features" in cfg:
        self._n_static_cols = int(cfg["n_static_features"])
        print(f"[static_feats] n_static_cols={self._n_static_cols} (from config.json n_static_features)")
        return self._n_static_cols

    # Fallback: load S.npz shape from FIDDLE output.
    if "data_dir" in cfg:
        data_dir = Path(cfg["data_dir"])
        task = cfg["task"]
        t1, t2 = cfg.get("theta_1", "0.01"), cfg.get("theta_2", "0.01")
        s_path = data_dir / "fiddle_output" / f"{task}_t1[{t1}]_t2[{t2}]" / "S.npz"
        if s_path.exists():
            s_data = np.load(s_path)
            n_static = int(s_data["shape"][1])
            self._n_static_cols = n_static
            print(f"[static_feats] n_static_cols={n_static} (from S.npz)")
            return n_static

    raise RuntimeError(
        f"Cannot determine n_static_cols from config at {cfg_path}. "
        f"Ensure config.json has 'n_static_features' or 'n_static_cols'."
    )


def _extract_static_features(self, batches):
    """Extract static features from eICU data.

    For TPC two-input format: grabs the static tensor directly.
    For legacy fused format: slices the last ``n_static_cols`` columns.

    Args:
        batches: List of (x, y) batches from DataLoader.

    Returns:
        Tensor [N, n_static] of static feature values.
    """
    # Detect format from first batch
    first_x = batches[0][0] if batches else None
    is_tuple = (isinstance(first_x, (list, tuple)) and len(first_x) == 2
                and all(torch.is_tensor(t) for t in first_x))

    all_static = []
    with torch.no_grad():
        for batch in batches:
            x, _ = process_batch(batch, self.device)
            if is_tuple:
                all_static.append(x[1].cpu())
            elif isinstance(x, torch.Tensor):
                if x.dim() == 3:
                    all_static.append(x[:, 0, -_get_n_static_cols(self):].cpu())
                else:
                    all_static.append(x[:, -_get_n_static_cols(self):].cpu())

    return torch.cat(all_static, dim=0).float()


_TPC_DEMOGRAPHIC_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]
"""Static-vector column indices for gender (0), age (1), and ethnicity (6–11) in TPC format."""


def _get_demographic_indices(self):
    """Return the column indices of demographic features within the static vector.

    For TPC two-input format the positions are stable and hardcoded.
    For FIDDLE fused format, ``generate_eICU.py`` saves them in ``config.json``
    as ``demographic_feature_indices``; falls back to the TPC defaults if not found.
    Result is cached on ``self``.
    """
    cached = getattr(self, "_demographic_indices", None)
    if cached is not None:
        return cached

    import json as _json
    from pathlib import Path

    dataset = getattr(self, "dataset", None) or getattr(self.args, "dataset", "")
    cfg_path = Path("..") / "dataset" / dataset / "config.json"
    try:
        with open(cfg_path) as f:
            cfg = _json.load(f)
        if "demographic_feature_indices" in cfg:
            self._demographic_indices = list(cfg["demographic_feature_indices"])
            return self._demographic_indices
    except Exception:
        pass

    self._demographic_indices = list(_TPC_DEMOGRAPHIC_INDICES)
    return self._demographic_indices


def _extract_demographic_features(self, batches):
    """Extract demographic feature columns from the static vector.

    Calls ``_extract_static_features`` then slices to the demographic indices
    returned by ``_get_demographic_indices``.

    Args:
        batches: List of (x, y) batches from DataLoader.

    Returns:
        Tensor [N, n_demo] of demographic feature values.
    """
    static_feats = _extract_static_features(self, batches)
    indices = _get_demographic_indices(self)
    max_idx = static_feats.size(1)
    valid_indices = [i for i in indices if i < max_idx]
    if not valid_indices:
        return static_feats
    return static_feats[:, valid_indices]


def preprocess_for_encoder(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.dim() == 4 and x.size(2) != 224:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, -1, 1, 1)
    return (x - mean) / std


def init_graph_encoder(name: str = "resnet18"):
    if name != "resnet18":
        raise NotImplementedError(f"Encoder '{name}' is not supported yet.")
    try:
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        base = models.resnet18(pretrained=True)
    encoder = torch.nn.Sequential(*list(base.children())[:-1])
    return encoder


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_train_eval_graph(
    self,
    tr_ds: torch.Tensor,
    tr_preds: torch.Tensor,
    tr_meta_labels: torch.Tensor,
    y_tr: torch.Tensor,
    tr_feats: torch.Tensor,
    eval_ds: torch.Tensor,
    y_eval: torch.Tensor,
    eval_feats: torch.Tensor,
    eval_type: str = "val",
    tr_edge_feats: torch.Tensor = None,
    eval_edge_feats: torch.Tensor = None,
) -> HeteroData:
    """
    Constructs the Heterogeneous Graph for FedDES (Train + Eval).
    """
    num_classifiers = tr_meta_labels.shape[1]

    # 1. PREPARE COMBINED DATA
    combined_ds, combined_y, combined_feats, combined_meta, n_train, n_total = (
        _prepare_combined_data(tr_ds, tr_preds, tr_meta_labels, y_tr, tr_feats,
                               eval_ds, y_eval, eval_feats)
    )

    # Prepare combined edge feats (may differ from node feats).
    if tr_edge_feats is not None and eval_edge_feats is not None:
        combined_edge_feats = torch.cat([_to_cpu(tr_edge_feats), _to_cpu(eval_edge_feats)], dim=0)
    else:
        combined_edge_feats = combined_feats

    # 2. BUILD EDGES
    ss_edge_index, ss_weights, cs_edge_index, cs_weights = _build_edges(
        self, combined_ds, combined_edge_feats, combined_y, combined_meta,
        n_train, n_total, num_classifiers,
    )

    # 3. CLASSIFIER NODE FEATURES
    clf_x = _compute_classifier_features(
        self, tr_ds, tr_meta_labels, y_tr, num_classifiers,
    )

    # 4. ASSEMBLE HETERODATA
    data = _assemble_hetero_data(
        combined_feats, combined_y, clf_x, num_classifiers,
        ss_edge_index, ss_weights, cs_edge_index, cs_weights,
        n_train, n_total, eval_type,
    )

    return data


# ---------------------------------------------------------------------------
# Sub-functions for build_train_eval_graph
# ---------------------------------------------------------------------------

def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.asarray(t)


def _to_cpu(t):
    if isinstance(t, torch.Tensor):
        return t.cpu()
    return torch.from_numpy(t)


def _to_tensor(x, dtype_fn):
    """Convert Numpy or Tensor to the desired dtype."""
    if isinstance(x, torch.Tensor):
        return dtype_fn(x)
    return dtype_fn(torch.from_numpy(x))


def _prepare_combined_data(
    tr_ds, tr_preds, tr_meta_labels, y_tr, tr_feats,
    eval_ds, y_eval, eval_feats,
):
    """Stack train + eval data, build combined meta-labels with eval dummies."""
    tr_ds_np = _to_np(tr_ds)
    eval_ds_np = _to_np(eval_ds)
    y_tr_np = _to_np(y_tr)
    y_eval_np = _to_np(y_eval)

    combined_ds = np.vstack([tr_ds_np, eval_ds_np])
    combined_y = np.concatenate([y_tr_np, y_eval_np])
    combined_feats = torch.cat([_to_cpu(tr_feats), _to_cpu(eval_feats)], dim=0)

    num_classifiers = tr_meta_labels.shape[1]
    n_eval = eval_ds.shape[0]
    eval_meta_dummy = torch.zeros((n_eval, num_classifiers), dtype=torch.float32)
    combined_meta = torch.cat([_to_cpu(tr_meta_labels).float(), eval_meta_dummy], dim=0)

    n_train = tr_ds.shape[0]
    n_total = combined_ds.shape[0]

    return combined_ds, combined_y, combined_feats, combined_meta, n_train, n_total


def _build_edges(self, combined_ds, combined_edge_feats, combined_y, combined_meta, n_train, n_total, num_classifiers):
    """Build SS and CS edges."""
    # A. Sample-Sample Edges (CMDW)
    # combined_edge_feats is the pre-resolved representation for edge construction
    # (controlled by graph_sample_edge_feats, independent of graph_sample_node_feats).
    edge_feat_mode = str(getattr(self.args, "graph_sample_edge_feats", "ds")).lower()
    if edge_feat_mode == "ds":
        ss_edge_matrix = combined_ds
    else:
        feats_np = combined_edge_feats.numpy() if hasattr(combined_edge_feats, "numpy") else np.array(combined_edge_feats)
        ss_edge_matrix = feats_np

    ss_edge_index, ss_weights = build_ss_edges_cmdw(
        decision_matrix=ss_edge_matrix,
        label_vector=combined_y,
        source_indices=np.arange(n_train, dtype=np.int64),
        destination_indices=np.arange(n_total, dtype=np.int64),
        k_per_class=int(getattr(self.args, "graph_k_per_class", 5)),
    )

    # B. Classifier-Sample Edges
    cs_mode = str(getattr(self.args, "graph_cs_mode", "balanced_acc:logloss")).lower()
    if ":" in cs_mode:
        score_mode, tie_break_mode = (part.strip() for part in cs_mode.split(":", 1))
    else:
        score_mode, tie_break_mode = cs_mode.strip(), "logloss"
    if tie_break_mode in {"none", ""}:
        tie_break_mode = None

    cs_topk = int(getattr(self.args, "graph_cs_topk", 3))
    if cs_topk == 0:
        cs_topk = max(1, int(0.25 * num_classifiers))

    cs_edge_index, cs_weights = build_cs_edges(
        tr_meta_labels=combined_meta.numpy(),
        decision_all=combined_ds,
        y_train=combined_y[:n_train],
        ss_edge_index=ss_edge_index,
        ss_edge_attr=ss_weights,
        n_train=n_train,
        n_total=n_total,
        num_classes=self.args.num_classes,
        top_k=cs_topk,
        score_mode=score_mode,
        tie_break_mode=tie_break_mode,
    )

    return ss_edge_index, ss_weights, cs_edge_index, cs_weights


def _compute_classifier_features(self, tr_ds, tr_meta_labels, y_tr, num_classifiers):
    """Compute per-class performance stats as classifier node features."""
    tr_ds_np = _to_np(tr_ds)
    y_tr_np = _to_np(y_tr)
    tr_meta_np = _to_np(tr_meta_labels).astype(np.float32)
    probs = tr_ds_np.reshape(-1, num_classifiers, self.args.num_classes)

    class_ids = np.unique(y_tr_np)
    class_masks = [y_tr_np == cls for cls in class_ids]
    class_counts = np.array([mask.sum() for mask in class_masks], dtype=np.float32)
    has_support = class_counts > 0
    safe_counts = np.where(has_support, class_counts, 1.0)[:, None]

    def masked_mean(values: np.ndarray) -> np.ndarray:
        out = []
        for mask, count in zip(class_masks, class_counts):
            if count > 0:
                out.append(values[mask].mean(axis=0))
            else:
                out.append(np.zeros(values.shape[1:], dtype=np.float32))
        return np.asarray(out, dtype=np.float32)

    per_class_hard_recall = masked_mean(tr_meta_np)
    true_class_probs = probs[np.arange(probs.shape[0]), :, y_tr_np][:, :, None]
    per_class_true_prob = masked_mean(true_class_probs.squeeze(-1))

    # Average margin per class
    probs_temp = probs.copy()
    rows = np.arange(probs.shape[0])
    probs_temp[rows[:, None], :, y_tr_np[:, None]] = -1.0
    max_rival_probs = probs_temp.max(axis=2)
    margins = true_class_probs.squeeze(-1) - max_rival_probs
    per_class_avg_margin = masked_mean(margins)

    se = np.sqrt(per_class_hard_recall * (1.0 - per_class_hard_recall) / safe_counts)
    se = np.where(has_support[:, None], se, 0.0).astype(np.float32)

    clf_x_stats = np.concatenate(
        [
            per_class_hard_recall.T,
            per_class_true_prob.T,
            per_class_avg_margin.T,
            se.T,
        ],
        axis=1,
    ).astype(np.float32)

    # Append per-class home-client data distribution (sums to 1 per classifier).
    dataset_name = getattr(self.args, "dataset", "")
    label_counts_map = load_client_label_counts(dataset_name)
    dist_feats = np.zeros((num_classifiers, self.args.num_classes), dtype=np.float32)
    for idx in range(num_classifiers):
        if idx < len(self.global_clf_keys):
            key = self.global_clf_keys[idx]
            home_role = key[0] if isinstance(key, (list, tuple)) and key else str(key)
        else:
            home_role = f"Client_{idx}"
        home_counts = label_counts_map.get(home_role, {})
        total = float(sum(home_counts.values()))
        if total > 0:
            for cls, cnt in home_counts.items():
                if 0 <= int(cls) < self.args.num_classes:
                    dist_feats[idx, int(cls)] = float(cnt) / total

    return torch.from_numpy(np.concatenate([clf_x_stats, dist_feats], axis=1)).float()


def _assemble_hetero_data(
    combined_feats, combined_y, clf_x, num_classifiers,
    ss_edge_index, ss_weights, cs_edge_index, cs_weights,
    n_train, n_total, eval_type,
):
    """Assemble the HeteroData graph object."""
    data = HeteroData()
    data['sample'].x = combined_feats.float()
    data['sample'].y = torch.from_numpy(combined_y).long()

    # Masks
    idx = torch.arange(n_total)
    data['sample'].train_mask = idx < n_train
    data['sample'][f"{eval_type}_mask"] = idx >= n_train

    data['classifier'].x = clf_x
    data['classifier'].num_nodes = num_classifiers

    # Edges
    data['sample', 'ss', 'sample'].edge_index = _to_tensor(ss_edge_index, lambda t: t.long())
    data['sample', 'ss', 'sample'].edge_attr = _to_tensor(ss_weights, lambda t: t.float())

    data['classifier', 'cs', 'sample'].edge_index = _to_tensor(cs_edge_index, lambda t: t.long())
    data['classifier', 'cs', 'sample'].edge_attr = _to_tensor(cs_weights, lambda t: t.float())

    # Empty CC edges (placeholder for future use)
    data['classifier', 'cc', 'classifier'].edge_index = torch.empty((2, 0), dtype=torch.long)
    data['classifier', 'cc', 'classifier'].edge_attr = torch.empty((0,), dtype=torch.float)

    return data

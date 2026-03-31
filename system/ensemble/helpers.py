from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def _safe_roc_auc(y_true: torch.Tensor, probs: torch.Tensor, num_classes: int) -> float:
    """Compute ROC AUC, returning NaN if undefined (e.g. single class in y)."""
    try:
        y_np = y_true.cpu().numpy()
        p_np = probs.cpu().float().numpy()
        if num_classes == 2:
            return float(roc_auc_score(y_np, p_np[:, 1]))
        else:
            return float(roc_auc_score(y_np, p_np, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        return float("nan")

def available_devices() -> List[str]:
    if torch.cuda.is_available():
        try:
            cuda_devices = torch.cuda.device_count()
            # Touch each device to ensure it is actually usable; otherwise fall back to CPU.
            for idx in range(cuda_devices):
                _ = torch.cuda.get_device_properties(idx)
            if cuda_devices:
                return [f"cuda:{idx}" for idx in range(cuda_devices)]
        except Exception as exc:
            print(f"[FedDES][Server] CUDA unavailable ({exc}); trying MPS/CPU.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]

def run_stage(clients: List[Any], stage: str, stage_inputs: Dict[str, Any] | Callable[[str], Dict[str, Any]] | None = None) -> None:
    """
    Run a named method across clients, optionally in parallel.

    Parallelism modes:
      - GPU: ThreadPoolExecutor across CUDA devices.
      - CPU with cpu_workers > 1: ThreadPoolExecutor with N threads.
        PyTorch releases the GIL during tensor ops (matmul, backward, etc.),
        so threads give real parallelism for the compute-heavy GNN training
        with zero extra memory overhead.
      - CPU with cpu_workers == 1: sequential (default).
    """
    import os

    devices = available_devices()
    n_gpu_workers = len(devices) if devices and devices[0] != "cpu" else 0

    # Determine CPU parallelism from the first client's args.
    cpu_workers_cfg = 1
    if clients and hasattr(clients[0], "args"):
        cpu_workers_cfg = int(getattr(clients[0].args, "cpu_workers", 1))
        if cpu_workers_cfg == 0:
            cpu_workers_cfg = max(1, (os.cpu_count() or 1) // 4)

    # Use CPU thread parallelism when on CPU, workers > 1, and multiple clients.
    use_cpu_parallel = (
        n_gpu_workers == 0
        and cpu_workers_cfg > 1
        and len(clients) > 1
    )

    if use_cpu_parallel:
        n_workers = min(cpu_workers_cfg, len(clients))
    elif n_gpu_workers > 0:
        n_workers = n_gpu_workers
    else:
        n_workers = 1

    print(f"[FedDES][run_stage] Stage={stage} devices={devices} "
          f"n_workers={n_workers} n_clients={len(clients)}"
          + (f" (cpu_threads)" if use_cpu_parallel else ""))

    def _move_input(value: Any, device: str, stage: str) -> Any:
        if isinstance(value, torch.nn.Module):
            cloned = copy.deepcopy(value)
            cloned.eval()
            if stage == "prepare_graph_data":
                return cloned.to("cpu")
            return cloned.to(device)
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _move_input(v, device, stage) for k, v in value.items()}
        if isinstance(value, list):
            return [_move_input(v, device, stage) for v in value]
        if isinstance(value, tuple):
            return tuple(_move_input(v, device, stage) for v in value)
        return value

    def _prepare_stage_kwargs(stage_inputs: Any | None, device: str, stage: str) -> Dict[str, Any]:
        if stage_inputs is None:
            return {}
        inputs = stage_inputs(device) if callable(stage_inputs) else stage_inputs
        if not isinstance(inputs, dict):
            raise TypeError("stage_inputs must be a dict or callable returning a dict")
        return {k: _move_input(v, device, stage) for k, v in inputs.items()}

    def run(client, device):
        fn = getattr(client, stage)
        kwargs = _prepare_stage_kwargs(stage_inputs, device, stage)
        print(f"[FedDES][run_stage] Client={client.role} Stage={stage} Device={device} Inputs={list(kwargs.keys())}")
        return fn(device, **kwargs)

    # --- Sequential path ---
    if n_workers <= 1:
        device = devices[0] if devices else "cpu"
        for client in clients:
            run(client, device)
        return

    # --- Thread-parallel path (GPU or CPU) ---
    device = devices[0] if devices else "cpu"

    if use_cpu_parallel:
        # Limit intra-op parallelism to avoid oversubscription.
        # Use NSLOTS (SGE) or cpu_count, divided by number of workers.
        total_threads = int(os.environ.get("NSLOTS", 0)) or os.cpu_count() or 1
        threads_per_worker = max(1, total_threads // n_workers)
        orig_threads = torch.get_num_threads()
        torch.set_num_threads(threads_per_worker)
        print(f"[FedDES][Server] Running {stage} with {n_workers} CPU threads "
              f"({threads_per_worker} intra-op threads each).")
    else:
        print(f"[FedDES][Server] Running {stage} with {n_workers} GPU workers.")

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(run, client, devices[idx % len(devices)] if n_gpu_workers else device): client
            for idx, client in enumerate(clients)
        }
        for fut, client in futures.items():
            try:
                fut.result()
            except Exception as exc:
                import traceback
                print(f"[FedDES][run_stage][ERROR] Client={client.role} Stage={stage} Exception={exc}")
                print(traceback.format_exc())
                raise

    if use_cpu_parallel:
        torch.set_num_threads(orig_threads)

# ---------------------------------------------------------------------------
# Fingerprint functions — canonical home is utils.fingerprinting.
# Re-exported here so existing ``from ensemble.helpers import derive_config_ids``
# continues to work.
# ---------------------------------------------------------------------------

from utils.fingerprinting import (  # noqa: F401
    config_fingerprint,
    derive_config_ids,
    derive_base_id_for_override,
    FP_EXCLUDE_IF_DEFAULT as _FP_EXCLUDE_IF_DEFAULT,
)


_MODEL_FAMILY_MODELS: Dict[str, List[str]] = {
    "HtFE-img-4": [
        "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)",
        "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
        "mobilenet_v2(pretrained=False, num_classes=args.num_classes)",
        "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
    ],
    "HtFE-img-5": [
        "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
        "mobilenet_v2(pretrained=False, num_classes=args.num_classes)",
        "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
        "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
        "torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)",
    ],
    # eICU TPC families — model strings use placeholder args.* values so that
    # fingerprints computed at config-expansion time (run_experiments.py) match
    # the fingerprints computed at runtime (clientDES.__init__).
    "eICU_tpc": [
        "RNN_V2(input_size=174, output_size=args.num_classes, hidden_size=64, num_layers=1, n_neurons=64, dropout=0.3, activation=\"relu\", static_dim=args.eicu_static_dim)",
        "RNN_V2(input_size=174, output_size=args.num_classes, hidden_size=32, num_layers=1, n_neurons=64, dropout=0.3, activation=\"relu\", static_dim=args.eicu_static_dim)",
        "TCN(in_channels=174, n_classes=args.num_classes, n_filters=32, kernel_size=3, num_levels=2, dropout=0.3, static_dim=args.eicu_static_dim)",
    ],
    "eICU_tpc_mixed3": [
        "RNN_V2(input_size=174, output_size=args.num_classes, hidden_size=64, num_layers=1, n_neurons=64, dropout=0.3, activation=\"relu\", static_dim=args.eicu_static_dim)",
        "TCN(in_channels=174, n_classes=args.num_classes, n_filters=32, kernel_size=3, num_levels=2, dropout=0.3, static_dim=args.eicu_static_dim)",
        "XGBoost()",
    ],
    "eICU_tpc_mixed5": [
        "RNN_V2(input_size=174, output_size=args.num_classes, hidden_size=64, num_layers=1, n_neurons=64, dropout=0.3, activation=\"relu\", static_dim=args.eicu_static_dim)",
        "RNN_V2(input_size=174, output_size=args.num_classes, hidden_size=32, num_layers=1, n_neurons=64, dropout=0.3, activation=\"relu\", static_dim=args.eicu_static_dim)",
        "TCN(in_channels=174, n_classes=args.num_classes, n_filters=32, kernel_size=3, num_levels=2, dropout=0.3, static_dim=args.eicu_static_dim)",
        "TCN(in_channels=174, n_classes=args.num_classes, n_filters=16, kernel_size=3, num_levels=2, dropout=0.3, static_dim=args.eicu_static_dim)",
        "XGBoost()",
        "RandomForest()",
    ],
    "eICU_tpc3": [
        "RNN_V2(input_size=174, output_size=args.num_classes, hidden_size=64, num_layers=1, n_neurons=64, dropout=0.3, activation=\"relu\", static_dim=args.eicu_static_dim)",
        "TCN(in_channels=174, n_classes=args.num_classes, n_filters=32, kernel_size=3, num_levels=2, dropout=0.3, static_dim=args.eicu_static_dim)",
        "ResNet1D(in_channels=174, base_filters=32, n_blocks=3, downsample_gap=3, increasefilter_gap=4, n_classes=args.num_classes, use_bn=True, use_do=True, dropout=0.3, static_dim=args.eicu_static_dim)",
    ],
}


def get_model_family_models(model_family: str) -> List[str] | None:
    """Return model strings for the given family, or None if unknown."""
    return _MODEL_FAMILY_MODELS.get(model_family)


def load_base_clf(client: Any, model_str: str):
    """Load a saved base classifier.

    For PyTorch models (``.pt``), loads and sets to eval mode.
    For tree models (``.pkl``), loads via joblib.

    In pool mode, loads from per-model directories using hospital ID.
    Falls back to legacy paths if pool-mode paths don't exist.
    """
    from ensemble.base_clf_utils import is_tree_model

    # model_str is a checkpoint name like "model_2"; resolve the actual
    # model definition (e.g. "XGBoost()") to determine the model type.
    model_id = int(model_str.split("_")[-1])
    model_def = client.args.models[model_id]

    # Use _ckpt_path if available (pool mode aware)
    if hasattr(client, "_ckpt_path"):
        model_path = client._ckpt_path(model_id, model_str)
    else:
        ext = "pkl" if is_tree_model(model_def) else "pt"
        model_path = Path(client.base_dir) / f"{client.role}_{model_str}.{ext}"

    if not model_path.exists():
        # Fallback to legacy path
        ext = "pkl" if is_tree_model(model_def) else "pt"
        model_path = Path(client.base_dir) / f"{client.role}_{model_str}.{ext}"

    if not model_path.exists():
        raise FileNotFoundError(f"Base classifier not found at {model_path}")

    if is_tree_model(model_def):
        import joblib
        return joblib.load(model_path)

    model = torch.load(model_path, map_location=client.device)
    model.eval()
    return model



def init_base_meta_loaders(client: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Split the client's train dataset into base/meta loaders (50/50).
    Caches loaders on the client to avoid rebuilding.
    """
    if getattr(client, "_base_train_loader", None) is not None and getattr(client, "_meta_train_loader", None) is not None:
        return client._base_train_loader, client._meta_train_loader

    # client.load_train_data only accepts batch_size; shuffling handled via random_split below.
    full_loader = client.load_train_data(batch_size=client.batch_size)
    dataset = full_loader.dataset
    n_total = len(dataset)
    g = torch.Generator().manual_seed(getattr(client.args, "base_split_seed", 0))
    n_base = n_total // 2
    n_meta = n_total - n_base
    base_ds, meta_ds = random_split(dataset, [n_base, n_meta], generator=g)

    seed = int(getattr(client.args, "seed", getattr(client.args, "base_split_seed", 0)))
    worker_init = lambda wid: np.random.seed(seed + wid)
    gen = torch.Generator().manual_seed(seed)

    client._base_train_loader = DataLoader(
        base_ds, batch_size=client.batch_size, shuffle=True, drop_last=True,
        worker_init_fn=worker_init, generator=gen,
    )
    client._meta_train_loader = DataLoader(
        meta_ds, batch_size=client.batch_size, shuffle=False, drop_last=False,
        worker_init_fn=worker_init, generator=gen,
    )
    return client._base_train_loader, client._meta_train_loader


def get_kfold_loaders(
    client: Any,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader, np.ndarray]]:
    """
    Build deterministic K-fold (train_loader, val_loader, val_idx) tuples
    over the full local training set. Val loaders do NOT shuffle.
    """
    full_train_loader = client.load_train_data(batch_size=client.batch_size)
    dataset = full_train_loader.dataset

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    worker_init = lambda wid: np.random.seed(seed + wid)

    loaders = []
    for train_idx, val_idx in kf.split(indices):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        gen = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            train_subset,
            batch_size=client.batch_size,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_init,
            generator=gen,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=client.batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_init,
            generator=gen,
        )
        loaders.append((train_loader, val_loader, val_idx))

    return loaders


def get_performance_baselines(client: Any, test_bundle: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Compute baseline global/local ensemble metrics (soft and hard) from cached test preds/y.
    Saves a nested dict keyed by "soft" and "hard".
    """
    test_preds = test_bundle["preds"]  # [N, M]
    y_test = test_bundle["y"]          # [N]
    ds = test_bundle["ds"]             # [N, M*C] flattened probs

    C = client.args.num_classes
    probs = ds.view(test_preds.size(0), test_preds.size(1), C).float()  # [N, M, C]

    def combine_soft(prob_tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return prob_tensor.mean(dim=1)
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (prob_tensor * weights.unsqueeze(-1)).sum(dim=1) / denom

    def combine_hard(pred_tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        one_hot = F.one_hot(pred_tensor, num_classes=C).float()  # [N, M, C]
        if mask is not None:
            one_hot = one_hot * mask.unsqueeze(-1)
        return one_hot.sum(dim=1).argmax(dim=1)

    key_to_idx = {k: i for i, k in enumerate(client.global_clf_keys)}
    local_indices = [key_to_idx[k] for k in client.local_clf_keys if k in key_to_idx]
    local_mask = torch.zeros_like(test_preds, dtype=torch.float)
    local_mask[:, local_indices] = 1.0

    metrics: Dict[str, Dict[str, float]] = {}

    # Soft
    global_probs = combine_soft(probs, None)
    local_probs = combine_soft(probs, local_mask if local_indices else None)
    global_preds = global_probs.argmax(dim=1)
    local_preds = local_probs.argmax(dim=1)
    metrics["soft"] = {
        "global_acc": (global_preds == y_test).float().mean().item(),
        "global_bacc": client.balanced_accuracy(global_preds, y_test),
        "global_auc": _safe_roc_auc(y_test, global_probs, C),
        "local_acc": (local_preds == y_test).float().mean().item(),
        "local_bacc": client.balanced_accuracy(local_preds, y_test),
        "local_auc": _safe_roc_auc(y_test, local_probs, C),
    }

    # Hard — AUC uses soft probs from the hard-voting ensemble (one-hot weighted)
    global_preds_h = combine_hard(test_preds, None)
    local_preds_h = combine_hard(test_preds, local_mask)
    global_hard_probs = F.one_hot(test_preds, num_classes=C).float().mean(dim=1)
    local_hard_probs = (F.one_hot(test_preds, num_classes=C).float() * local_mask.unsqueeze(-1)).sum(dim=1)
    local_hard_denom = local_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    local_hard_probs = local_hard_probs / local_hard_denom
    metrics["hard"] = {
        "global_acc": (global_preds_h == y_test).float().mean().item(),
        "global_bacc": client.balanced_accuracy(global_preds_h, y_test),
        "global_auc": _safe_roc_auc(y_test, global_hard_probs, C),
        "local_acc": (local_preds_h == y_test).float().mean().item(),
        "local_bacc": client.balanced_accuracy(local_preds_h, y_test),
        "local_auc": _safe_roc_auc(y_test, local_hard_probs, C),
    }

    # Voting baseline treated same as hard voting (unweighted subset)
    metrics["voting"] = metrics["hard"].copy()

    individual_classifier_perf: List[Dict[str, Any]] = []
    for idx, clf_key in enumerate(client.global_clf_keys):
        clf_preds = test_preds[:, idx]
        acc = (clf_preds == y_test).float().mean().item()
        bacc = client.balanced_accuracy(clf_preds, y_test)
        individual_classifier_perf.append({
            "classifier": f"{clf_key[0]}:{clf_key[1]}",
            "home_client": clf_key[0],
            "model": clf_key[1],
            "acc": acc,
            "bacc": bacc,
        })
    metrics["individual_classifier_perf"] = individual_classifier_perf

    out_path = Path(client.graph_dir) / f"{client.role}_performance_baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Dataset / partition ID utilities — canonical home is utils.partition_utils.
# Re-exported here so ``from ensemble.helpers import build_cifar_partition_id``
# continues to work inside the torch-enabled runtime.
# ---------------------------------------------------------------------------

from utils.partition_utils import (  # noqa: F401  (re-export)
    partition_cfg_hash,
    build_partition_dir_name,
    build_cifar_default_label,
    build_cifar_partition_id,
    build_eicu_partition_id,
    find_partition_by_hash,
    extract_cfg_hash,
    _legacy_dataset_partition_id,
    _legacy_eicu_partition_id,
)


def extract_eicu_task_from_dataset(dataset_name: str) -> str:
    marker = "task=["
    idx = dataset_name.find(marker)
    if idx == -1:
        return dataset_name
    idx += len(marker)
    end = dataset_name.find("]", idx)
    if end == -1:
        end = len(dataset_name)
    return dataset_name[idx:end]

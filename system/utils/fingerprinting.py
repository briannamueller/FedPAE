"""
Consolidated fingerprint / hash-ID module for HtFLlib.
======================================================

**This is the single canonical source** for all hash computation used to name
dataset-partition directories and experiment-artifact directories.

Three hash types:

1. ``partition_fingerprint``  — SHA-256, 6-char hex.
   Used for dataset partition directory names: ``{label}_cfg[{hash}]``.

2. ``config_fingerprint``     — MD5, 8-char hex.
   Used for experiment config directory names:
   ``base[X]``, ``graph[Y]``, ``gnn[Z]``, ``pae[P]``, ``hs[H]``, ``deslib[D]``.

3. ``pfl_fingerprint``        — MD5, 8-char hex.
   Used for PFL baseline result filenames: ``pfl[X]``.

Every other module should import from here (or from thin re-export wrappers
in ``partition_utils.py`` / ``des/helpers.py`` / ``pfl_fingerprint.py``).

Adding a new argument:
  - Partition params  → update the caller that builds the ``params`` dict
  - Experiment config → it's automatic (prefix-based key selection)
  - PFL config        → add the key to ``PFL_COMMON_KEYS`` or ``PFL_ALGO_KEYS``
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level hashing
# ---------------------------------------------------------------------------

def _hash(params: dict, *, algo: str = "md5", length: int = 8) -> str:
    """Deterministic hex hash of *params*.

    Parameters
    ----------
    params : dict
        Key/value pairs to hash. Serialised with ``json.dumps(sort_keys=True)``.
    algo : str
        ``"md5"`` or ``"sha256"``.
    length : int
        Number of hex characters to return.
    """
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    h = hashlib.new(algo, blob)
    return h.hexdigest()[:length]


# ===================================================================
# 1.  PARTITION FINGERPRINTS  (SHA-256, 6-char)
# ===================================================================

def partition_fingerprint(params: dict) -> str:
    """Deterministic 6-char SHA-256 hex hash of partition config *params*."""
    return _hash(params, algo="sha256", length=6)


# --- directory name construction ------------------------------------------

def build_partition_dir_name(cfg_hash: str, label: str = "") -> str:
    """``{label}_cfg[{hash}]`` or ``cfg[{hash}]``."""
    if label:
        safe_label = re.sub(r'[\s/\\]+', '_', label.strip()).strip('_')
        return f"{safe_label}_cfg[{cfg_hash}]"
    return f"cfg[{cfg_hash}]"


def build_cifar_default_label(partition_type: str, alpha: float, C: int) -> str:
    """Auto-generate a human-readable label for CIFAR partitions."""
    def _f(v):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return f"{v:.6g}" if isinstance(v, float) else str(v)

    if partition_type == "dir":
        return f"dir_alpha{_f(alpha)}"
    elif partition_type == "exdir":
        return f"exdir_alpha{_f(alpha)}_C{_f(C)}"
    elif partition_type == "pat":
        return f"pat_C{_f(C)}"
    return partition_type


def build_cifar_partition_id(
    partition_type: str,
    alpha: float,
    C: int,
    min_size: int,
    train_ratio: float,
    seed: int,
    num_clients: int,
    label: str = "",
) -> str:
    """Hash-based CIFAR partition ID: ``{label}_cfg[{hash}]``."""
    params = dict(
        partition_type=partition_type, alpha=alpha, C=C,
        min_size=min_size, train_ratio=train_ratio,
        seed=seed, num_clients=num_clients,
    )
    cfg_hash = partition_fingerprint(params)
    if not label:
        label = build_cifar_default_label(partition_type, alpha, C)
    return build_partition_dir_name(cfg_hash, label)


def build_eicu_partition_id(
    task: str,
    min_size: int,
    seed: int,
    train_ratio: float,
    num_clients: int,
    theta_1: str = "0.01",
    theta_2: str = "0.01",
    client_sort_mode: str = "prevalence",
    prefer_positive: bool = False,
    label: str = "",
    outer_kfold: Optional[int] = None,
    min_minority: int = 0,
    min_prev: float = 0.0,
    min_subgroup_samples: int = 0,
    subgroup_attr: str = "ethnicity",
    drop_hospital_vars: bool = False,
    fused: bool = False,
    include_diagnoses: bool = True,
    diag_window: str = "5h",
) -> str:
    """Hash-based eICU partition ID: ``{label}_cfg[{hash}]``.

    ``min_minority`` and ``min_prev`` are only included in the hash when
    they differ from their neutral defaults (0 and 0.0).  Similarly,
    ``min_subgroup_samples``, ``drop_hospital_vars``, ``fused``,
    ``include_diagnoses``, and ``diag_window`` are only included when
    non-default.  This preserves backward-compatibility with partition
    directories created before these parameters were tracked.
    """
    params = dict(
        task=task, min_size=min_size, num_clients=num_clients,
        train_ratio=train_ratio, seed=seed,
        theta_1=str(theta_1), theta_2=str(theta_2),
        client_sort_mode=client_sort_mode,
        prefer_positive=prefer_positive,
    )
    if outer_kfold is not None:
        params["outer_kfold"] = int(outer_kfold)
    # Legacy defaults: these values were used by existing partitions but not
    # included in the hash.  Only include them when they differ, so that old
    # partition hashes are preserved.
    if int(min_minority) != 25:
        params["min_minority"] = int(min_minority)
    if float(min_prev) != 0.05:
        params["min_prev"] = float(min_prev)
    if int(min_subgroup_samples) > 0:
        params["min_subgroup_samples"] = int(min_subgroup_samples)
        params["subgroup_attr"] = str(subgroup_attr)
    if drop_hospital_vars:
        params["drop_hospital_vars"] = True
    if fused:
        params["fused"] = True
    if not include_diagnoses:
        params["include_diagnoses"] = False
    if diag_window != "5h":
        params["diag_window"] = str(diag_window)
    cfg_hash = partition_fingerprint(params)
    if not label:
        label = task
    return build_partition_dir_name(cfg_hash, label)


def build_eicu_pool_partition_id(
    task: str,
    min_size: int,
    seed: int,
    train_ratio: float,
    theta_1: str = "0.01",
    theta_2: str = "0.01",
    client_sort_mode: str = "prevalence",
    prefer_positive: bool = False,
    label: str = "",
    outer_kfold: Optional[int] = None,
    drop_hospital_vars: bool = False,
    fused: bool = False,
    include_diagnoses: bool = True,
    diag_window: str = "5h",
) -> str:
    """Hash-based eICU *pool* partition ID for FedDES checkpoint reuse.

    Unlike :func:`build_eicu_partition_id`, this **excludes** filtering
    criteria that only select a subset of hospitals (min_minority, min_prev,
    min_subgroup_samples, subgroup_attr, num_clients).  This means all
    filter combinations share the same data pool on disk.
    """
    params = dict(
        task=task, min_size=min_size,
        train_ratio=train_ratio, seed=seed,
        theta_1=str(theta_1), theta_2=str(theta_2),
        client_sort_mode=client_sort_mode,
        prefer_positive=prefer_positive,
        pool_mode=True,  # distinguish from legacy partition IDs
    )
    if outer_kfold is not None:
        params["outer_kfold"] = int(outer_kfold)
    if drop_hospital_vars:
        params["drop_hospital_vars"] = True
    if fused:
        params["fused"] = True
    if not include_diagnoses:
        params["include_diagnoses"] = False
    if diag_window != "5h":
        params["diag_window"] = str(diag_window)
    cfg_hash = partition_fingerprint(params)
    if not label:
        label = task
    return build_partition_dir_name(cfg_hash, label)


# --- lookup helpers -------------------------------------------------------

def extract_cfg_hash(partition_id: str) -> Optional[str]:
    """Extract the 6-char hex hash from a string containing ``cfg[...]``."""
    m = re.search(r'cfg\[([0-9a-fA-F]{6})\]', partition_id)
    return m.group(1) if m else None


def find_by_hash(search_root: Path, cfg_hash: str) -> Optional[str]:
    """Scan *search_root* for a subdirectory whose name contains ``cfg[{hash}]``.

    Returns the directory *name* (not full path), or ``None``.
    """
    target = f"cfg[{cfg_hash}]"
    if not search_root.is_dir():
        return None
    for entry in search_root.iterdir():
        if entry.is_dir() and target in entry.name:
            return entry.name
    return None


def find_by_metadata(
    search_root: Path,
    target_params: dict,
    defaults: Optional[dict] = None,
) -> Optional[str]:
    """Scan *search_root* for a subdirectory whose ``_meta.json`` matches.

    Returns the directory *name* (not full path), or ``None``.
    """
    if not search_root.is_dir():
        return None
    for entry in search_root.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "_meta.json"
        if not meta_path.exists():
            continue
        try:
            stored = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stored_params = stored.get("params", {})
        if _params_match(stored_params, target_params, defaults or {}):
            return entry.name
    return None


def find_artifact(
    search_root: Path,
    params: dict,
    *,
    defaults: Optional[dict] = None,
    hash_func=None,
    tag_format: str = "cfg",
) -> Optional[str]:
    """Hash-first, metadata-fallback artifact lookup.

    Parameters
    ----------
    search_root : Path
        Directory to scan for matching subdirectories.
    params : dict
        The parameters that define the artifact.
    defaults : dict, optional
        Default values for params not present in stored metadata.
    hash_func : callable, optional
        Hash function to compute the expected hash.  Defaults to
        ``partition_fingerprint``.
    tag_format : str
        The bracket tag format in directory names, e.g. ``"cfg"`` matches
        ``cfg[ab12cd]``.

    Returns
    -------
    str or None
        Directory name if found, else ``None``.
    """
    if hash_func is None:
        hash_func = partition_fingerprint
    cfg_hash = hash_func(params)
    target = f"{tag_format}[{cfg_hash}]"

    # Fast path: hash-based lookup
    if search_root.is_dir():
        for entry in search_root.iterdir():
            if entry.is_dir() and target in entry.name:
                return entry.name

    # Slow path: metadata fallback
    return find_by_metadata(search_root, params, defaults)


def _params_match(
    stored: dict,
    requested: dict,
    defaults: dict,
) -> bool:
    """Check whether *stored* metadata params match *requested* params.

    - Keys in both must have equal values.
    - Keys in *requested* but missing from *stored* must equal the default
      (the param didn't exist when the artifact was created).
    - Keys in *stored* but missing from *requested* are ignored.
    """
    for k, v in requested.items():
        if k in stored:
            if _normalize_for_compare(stored[k]) != _normalize_for_compare(v):
                return False
        else:
            # Key missing from stored → must equal default
            default_val = defaults.get(k)
            if _normalize_for_compare(v) != _normalize_for_compare(default_val):
                return False
    return True


def _normalize_for_compare(v: Any) -> Any:
    """Normalize a value for comparison (NOT for hashing — hashing is unchanged)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if v is None:
        return None
    return v


# ===================================================================
# 2.  EXPERIMENT CONFIG FINGERPRINTS  (MD5, 8-char)
# ===================================================================

# Keys excluded from fingerprint when they equal their default value.
# Allows adding new args without breaking existing hashes, as long as
# the default preserves the original behaviour.
FP_EXCLUDE_IF_DEFAULT: Dict[str, Any] = {
    "graph_prune_metric": "minority_recall",
    "graph_prune_mode": "global",
    "graph_prune_lambda": 0,
    "graph_prune_threshold": 0,
    "gnn_balance_meta_elements": "none",
}


_EXCLUDE_FROM_BASE_FP = {"base_single_model"}


def _base_fp_from_data(data: Mapping[str, Any]) -> str:
    """Compute an 8-char MD5 fingerprint for the *base* config group.

    Special rules (must match ``make_phase_configs.fingerprint`` for ``prefix="base"``):
    - Collects all keys starting with ``"base"`` (except those in
      ``_EXCLUDE_FROM_BASE_FP`` — currently ``base_single_model``, which
      only affects pool composition, not individual model training)
    - Adds ``model_family`` unless it is ``"HtFE-img-5"`` or ``"eICU"``
    - Always adds ``feature_dim``
    """
    cfg = {k: v for k, v in data.items()
           if isinstance(k, str) and k.startswith("base") and k not in _EXCLUDE_FROM_BASE_FP}
    if "model_family" in data:
        mf = data["model_family"]
        if mf not in {"HtFE-img-5", "eICU"}:
            cfg["model_family"] = mf
    if "feature_dim" in data:
        cfg["feature_dim"] = data["feature_dim"]
    if not cfg:
        return "none"
    return _hash(cfg, algo="md5", length=8)


def config_fingerprint(data: Mapping[str, Any], prefix: str) -> str:
    """Compute an 8-char MD5 fingerprint for all args matching *prefix*.

    This is the canonical implementation.  ``make_phase_configs.fingerprint``
    and ``des.helpers._fp_for_prefix`` are thin wrappers / re-exports.

    Parameters
    ----------
    data : Mapping
        Full argument namespace (as dict) or ``vars(args)``.
    prefix : str
        One of ``"base"``, ``"graph"``, ``"gnn"``, ``"pae"``, ``"hs"``,
        ``"deslib"``.
    """
    if prefix == "base":
        return _base_fp_from_data(data)
    cfg = {}
    for k, v in data.items():
        if not (isinstance(k, str) and k.startswith(prefix)):
            continue
        if k in FP_EXCLUDE_IF_DEFAULT and v == FP_EXCLUDE_IF_DEFAULT[k]:
            continue
        cfg[k] = v
    # base_single_model affects pool composition, which is a graph-level concern.
    if prefix == "graph" and "base_single_model" in data:
        cfg["base_single_model"] = data["base_single_model"]
    if not cfg:
        return "none"
    return _hash(cfg, algo="md5", length=8)


def derive_config_ids(
    args: Any,
    prefixes: Tuple[str, ...] = ("base", "graph", "gnn"),
) -> Tuple[str, ...]:
    """Derive MD5 fingerprints for each prefix group."""
    data: Mapping[str, Any] = args if isinstance(args, Mapping) else vars(args)
    return tuple(config_fingerprint(data, p) for p in prefixes)


def derive_base_id_for_override(
    args: Any,
    model_family: str,
    models: list,
) -> str:
    """Compute base fingerprint with overridden model_family / models."""
    data = dict(vars(args))
    data["model_family"] = model_family
    data["models"] = models
    return _base_fp_from_data(data)


# --- Pool-mode per-model fingerprinting ------------------------------------

# Training hparams that affect individual model checkpoints.
_MODEL_TRAINING_KEYS = (
    "base_clf_lr", "base_es_metric", "base_es_patience", "base_es_min_delta",
    "base_weighted_by_class", "base_optimizer", "base_split_mode",
    "base_split_seed", "feature_dim", "local_epochs",
)


def extract_training_hparams(args: Any) -> dict:
    """Extract training hyperparams from *args* for per-model hashing."""
    data = args if isinstance(args, Mapping) else vars(args)
    return {k: data[k] for k in _MODEL_TRAINING_KEYS if k in data}


def model_checkpoint_hash(model_def_str: str, training_hparams: dict) -> str:
    """8-char MD5 hash identifying a single model's checkpoint.

    Combines the model definition string (e.g. ``"RNN_V2(...)")``
    with training hyperparams so that two identical model+hparam
    combos always produce the same hash — regardless of model family.
    """
    params = dict(training_hparams)
    params["_model_def"] = model_def_str
    return _hash(params, algo="md5", length=8)


def pool_level_fingerprint(
    model_hashes: list,
    hospital_ids: Optional[list] = None,
) -> str:
    """8-char MD5 hash combining per-model hashes and hospital selection.

    Used to name directories that depend on the full classifier pool
    (graph bundles, GNN artifacts) rather than individual models.

    When *hospital_ids* is provided, the hash also encodes which hospitals
    are in the selection, so different filter settings (min_minority etc.)
    produce different pool fingerprints and don't collide on disk.
    """
    params: dict = {"_pool": sorted(model_hashes)}
    if hospital_ids:
        params["_hospitals"] = sorted(hospital_ids)
    return _hash(params, algo="md5", length=8)


def selection_fingerprint(
    min_minority: int = 0,
    min_prev: float = 0.0,
    min_subgroup_samples: int = 0,
    subgroup_attr: str = "ethnicity",
    num_clients: int = 0,
    client_sort_mode: str = "prevalence",
) -> str:
    """6-char SHA-256 hash of client-selection criteria.

    Used to name the ``selection[{hash}]`` symlink directories inside
    a pool partition.
    """
    params: dict = {}
    if int(min_minority) != 0:
        params["min_minority"] = int(min_minority)
    if float(min_prev) != 0.0:
        params["min_prev"] = float(min_prev)
    if int(min_subgroup_samples) > 0:
        params["min_subgroup_samples"] = int(min_subgroup_samples)
        params["subgroup_attr"] = str(subgroup_attr)
    if int(num_clients) > 0:
        params["num_clients"] = int(num_clients)
    if client_sort_mode != "prevalence":
        params["client_sort_mode"] = client_sort_mode
    return _hash(params, algo="sha256", length=6)


# ===================================================================
# 3.  PFL BASELINE FINGERPRINTS  (MD5, 8-char)
# ===================================================================

PFL_COMMON_KEYS = frozenset({
    "model_family",
    "feature_dim",
    "num_classes",
    "local_learning_rate",
    "batch_size",
    "local_epochs",
    "global_rounds",
    "join_ratio",
    "learning_rate_decay",
    "learning_rate_decay_gamma",
    "weighted_loss",
    "auto_break",
    "use_val",
    "val_ratio",
    "split_seed",
    "global_model_idx",
})

PFL_ALGO_KEYS = frozenset({
    # FedProto / FD / FedTGP / FedKTL
    "lamda",
    # FedGen
    "noise_dim",
    "generator_learning_rate",
    "hidden_dim",
    # FedGen / FedGH / FedTGP / FedKTL
    "server_epochs",
    # FML
    "alpha",
    "beta",
    # FedKD
    "mentee_learning_rate",
    "T_start",
    "T_end",
    # FedGH
    "server_learning_rate",
    # FedTGP
    "margin_threthold",
    # FedKTL
    "server_batch_size",
    "mu",
    "generator_path",
    # FedMRL
    "sub_feature_dim",
})

PFL_KEYS = PFL_COMMON_KEYS | PFL_ALGO_KEYS


# Lazily cached argparse defaults — populated on first call.
_arg_defaults_cache: Optional[dict] = None


def _load_arg_defaults() -> dict:
    """Load argparse defaults from ``system/arg_parser.py``.

    Cached after first call.
    """
    global _arg_defaults_cache
    if _arg_defaults_cache is not None:
        return _arg_defaults_cache

    # Try to find arg_parser.py relative to this file
    parser_path = Path(__file__).resolve().parent.parent / "arg_parser.py"
    if not parser_path.exists():
        # Fallback: look relative to repo root
        parser_path = Path(__file__).resolve().parent.parent.parent / "system" / "arg_parser.py"
    if not parser_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("htfl_arg_parser", parser_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_arg_parser"):
        return {}
    parser = module.build_arg_parser()
    _arg_defaults_cache = vars(parser.parse_args([]))
    return _arg_defaults_cache


def pfl_fingerprint(args_or_dict: Any) -> str:
    """Return an 8-char MD5 hex fingerprint of PFL-relevant config.

    Fills missing keys from argparse defaults so that callers don't need
    to maintain their own ``_FINGERPRINT_DEFAULTS`` dict.

    Parameters
    ----------
    args_or_dict : argparse.Namespace | Mapping
        Full or partial argument namespace.  Only keys in ``PFL_KEYS``
        are included in the hash.

    Returns
    -------
    str
        8-char lowercase hex string, e.g. ``"a3f8c012"``.
    """
    data: Mapping[str, Any] = (
        args_or_dict if isinstance(args_or_dict, Mapping) else vars(args_or_dict)
    )
    defaults = _load_arg_defaults()

    # Build the fingerprint dict: defaults first, then overlay provided values.
    cfg = {}
    for k in PFL_KEYS:
        if k in data:
            cfg[k] = data[k]
        elif k in defaults:
            cfg[k] = defaults[k]
    if not cfg:
        return "none"
    return _hash(cfg, algo="md5", length=8)


# ===================================================================
# Legacy builders (kept for migration scripts only)
# ===================================================================

def _legacy_dataset_partition_id(
    partition_type: str, alpha: float, C: int, min_size: int,
    train_ratio: float, seed: int, num_clients: int,
) -> str:
    """Legacy two-level format: ``nc[N]_tr[R]_s[S]/exdir[alpha=A,C=C]``."""
    def _fmt(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    components = []
    if partition_type == "pat":
        components.append(f"C={_fmt(C)}")
    elif partition_type == "dir":
        components.append(f"alpha={_fmt(alpha)}")
    elif partition_type == "exdir":
        components.append(f"alpha={_fmt(alpha)}")
        components.append(f"C={_fmt(C)}")
    detail = ",".join(components)
    prefix = f"nc[{num_clients}]_tr[{_fmt(train_ratio)}]_s[{seed}]"
    return f"{prefix}/{partition_type}[{detail}]"


def _legacy_eicu_partition_id(
    task: str, min_size: int, seed: int, train_ratio: float,
    num_clients: int, theta_1: str = "0.01", theta_2: str = "0.01",
) -> str:
    """Legacy two-level format: ``min_size[M]_nc[N]_tr[R]_s[S]/task[T]_t1[X]_t2[Y]``."""
    def _fmt(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    prefix = f"min_size[{min_size}]_nc[{num_clients}]_tr[{_fmt(train_ratio)}]_s[{seed}]"
    detail = f"task[{task}]_t1[{theta_1}]_t2[{theta_2}]"
    return f"{prefix}/{detail}"

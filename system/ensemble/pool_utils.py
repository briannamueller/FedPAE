"""Shared pool-mode utilities for FedPAE and FedDES.

Pool mode allows a large pre-generated dataset of all qualifying hospitals to
be filtered at runtime by minimum sample counts, prevalence thresholds, and
subgroup constraints, rather than regenerating data for each filter setting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch

from ensemble.client_selection import (
    select_clients,
    ensure_selection_dir,
    _create_fold_symlinks,
    _write_selection_config,
)
from utils.fingerprinting import selection_fingerprint


def setup_pool_mode(server, args) -> None:
    """Configure pool-mode client selection and symlink directories.

    Reads the pool-mode config.json, applies filter criteria to select
    qualifying hospitals, creates a symlink-based selection directory so
    that downstream data loading can use contiguous integer indices, and
    updates ``args.dataset``, ``args.hospital_ids``, and ``args.num_clients``.

    If the dataset is not pool-mode or no hospitals pass the filters,
    ``args.pool_mode`` is set to False and the function returns early.
    """
    dataset_root = Path("..") / "dataset"

    # args.dataset may include a fold suffix, e.g.
    # "eICU_tpc/mortality_24h_cfg[abc123]/fold_0"
    pool_dataset_base = args.dataset
    fold_suffix = ""
    parts = args.dataset.split("/")
    if len(parts) >= 2 and parts[-1].startswith("fold_"):
        fold_suffix = parts[-1]
        pool_dataset_base = "/".join(parts[:-1])

    pool_data_dir = dataset_root / pool_dataset_base
    pool_config_path = pool_data_dir / "config.json"

    if not pool_config_path.exists():
        print(f"[Pool] No pool config at {pool_config_path}, falling back to standard mode")
        args.pool_mode = False
        return

    with open(pool_config_path) as f:
        pool_config = json.load(f)

    if pool_config.get("mode") != "pool":
        print("[Pool] Config is not pool-mode, falling back to standard mode")
        args.pool_mode = False
        return

    hospital_ids = select_clients(
        pool_config,
        min_minority=int(getattr(args, "min_minority", 0)),
        min_prev=float(getattr(args, "min_prev", 0.0)),
        min_subgroup_samples=int(getattr(args, "min_subgroup_samples", 0)),
        subgroup_attr=str(getattr(args, "subgroup_attr", "ethnicity")),
        num_clients=int(getattr(args, "num_clients", 0) or 0),
        client_sort_mode=str(getattr(args, "client_sort_mode", "prevalence")),
    )

    if not hospital_ids:
        raise ValueError("[Pool] No hospitals pass the filter criteria")

    sel_hash = selection_fingerprint(
        min_minority=int(getattr(args, "min_minority", 0)),
        min_prev=float(getattr(args, "min_prev", 0.0)),
        min_subgroup_samples=int(getattr(args, "min_subgroup_samples", 0)),
        subgroup_attr=str(getattr(args, "subgroup_attr", "ethnicity")),
        num_clients=int(getattr(args, "num_clients", 0) or 0),
        client_sort_mode=str(getattr(args, "client_sort_mode", "prevalence")),
    )

    if fold_suffix:
        sel_dir_name = f"selection[{sel_hash}]"
        fold_dir = pool_data_dir / fold_suffix
        _create_fold_symlinks(fold_dir, sel_dir_name, hospital_ids)
        _write_selection_config(
            fold_dir / sel_dir_name / "config.json",
            pool_config, hospital_ids,
            fold_idx=int(fold_suffix.split("_")[1]),
        )
        args.dataset = f"{pool_dataset_base}/{fold_suffix}/{sel_dir_name}"
    else:
        sel_dir_name = ensure_selection_dir(
            pool_data_dir, hospital_ids, sel_hash,
            pool_config, outer_kfold=None,
        )
        args.dataset = f"{pool_dataset_base}/{sel_dir_name}"

    args.hospital_ids = hospital_ids
    args.num_clients = len(hospital_ids)
    server.num_clients = len(hospital_ids)
    server.num_join_clients = int(server.num_clients * server.join_ratio)
    server.dataset = args.dataset

    print(f"[Pool] Selected {len(hospital_ids)} hospitals, selection[{sel_hash}]")
    print(f"[Pool] Dataset path: {args.dataset}")


def slice_graph_bundle(
    bundle: dict,
    keep_indices: List[int],
    M_orig: int,
    num_classes: int,
    feat_mode: str,
    feats_by_split: dict = None,
) -> tuple:
    """Slice a graph bundle to keep only selected classifier columns.

    Args:
        bundle: dict with keys ``"train"``, ``"val"``, ``"test"``, each
            containing ``ds [N, M*C]``, ``preds [N, M]``, ``meta [N, M]``,
            ``y [N]``.
        keep_indices: classifier indices to retain (0-based into *M_orig*).
        M_orig: original number of classifiers.
        num_classes: number of output classes C.
        feat_mode: ``graph_sample_node_feats`` value (``"ds"``,
            ``"embedding_concat"``, ``"embedding_mean"``, etc.).
        feats_by_split: dict mapping split names to feat tensors ``[N, F]``.

    Returns:
        ``(sliced_bundle, sliced_feats)`` where both are dicts keyed by split.
    """
    keep = torch.tensor(keep_indices, dtype=torch.long)
    sliced: dict = {}
    sliced_feats: dict = {}

    for split_name in ("train", "val", "test"):
        if split_name not in bundle:
            continue
        d = bundle[split_name]
        N = d["ds"].size(0)

        # ds: [N, M*C] -> [N, M, C] -> select -> [N, M_new*C]
        probs = d["ds"].view(N, M_orig, num_classes)
        ds_new = probs[:, keep, :].reshape(N, -1)

        preds_new = d["preds"][:, keep]
        meta_new = d["meta"][:, keep]
        y = d["y"]

        feats = feats_by_split[split_name] if feats_by_split else None
        feat_mode_lc = feat_mode.lower()

        if feats is None or feat_mode_lc in ("embedding_mean", "encoder", "demographics"):
            feats_new = feats
        elif feat_mode_lc == "ds":
            feats_new = ds_new
        elif feat_mode_lc == "hybrid":
            enc_dim = feats.size(1) - M_orig * num_classes
            feats_new = torch.cat([feats[:, :enc_dim], ds_new], dim=1) if enc_dim > 0 else ds_new
        elif feat_mode_lc == "embedding_concat":
            embed_dim = feats.size(1) // M_orig
            if embed_dim * M_orig == feats.size(1):
                feats_new = feats.view(N, M_orig, embed_dim)[:, keep, :].reshape(N, -1)
            else:
                feats_new = feats
        elif feat_mode_lc == "hybrid_demographics":
            demo_dim = feats.size(1) - M_orig * num_classes
            feats_new = torch.cat([ds_new, feats[:, M_orig * num_classes:]], dim=1) if demo_dim > 0 else ds_new
        else:
            feats_new = feats

        sliced[split_name] = {
            "ds": ds_new,
            "preds": preds_new,
            "meta": meta_new.float(),
            "y": y,
        }
        sliced_feats[split_name] = feats_new

    return sliced, sliced_feats

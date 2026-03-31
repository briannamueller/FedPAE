"""Runtime client selection for FedDES pool-mode data.

Reads pool metadata (hospital stats) from config.json and applies filtering
criteria to select a subset of hospitals.  Creates lightweight selection
directories with symlinks so that shared data-loading code (which uses
positional indices) works unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.fingerprinting import selection_fingerprint


# ---------------------------------------------------------------------------
# Client selection
# ---------------------------------------------------------------------------

def select_clients(
    pool_config: dict,
    min_minority: int = 0,
    min_prev: float = 0.0,
    min_subgroup_samples: int = 0,
    subgroup_attr: str = "ethnicity",
    num_clients: int = 0,
    client_sort_mode: str = "prevalence",
) -> List[int]:
    """Select hospitals from pool metadata that pass all filter criteria.

    Parameters
    ----------
    pool_config : dict
        Parsed config.json from a pool-mode partition.  Must contain
        ``hospital_metadata`` and ``all_hospital_ids``.
    min_minority, min_prev, min_subgroup_samples, subgroup_attr :
        Filtering criteria (same semantics as the data generation script).
    num_clients : int
        Max hospitals to keep (0 = all that qualify).
    client_sort_mode : str
        ``"prevalence"`` (positive rate) or ``"positives"`` (positive count).

    Returns
    -------
    list of int
        Ordered hospital IDs that pass all filters.
    """
    metadata = pool_config["hospital_metadata"]
    all_ids = pool_config["all_hospital_ids"]

    _ETHNICITY_FOCUS = {"Caucasian", "African American"}

    selected: List[Tuple[int, float]] = []
    for hid in all_ids:
        stats = metadata[str(hid)]
        n_pos = stats["n_positive"]
        n_neg = stats["n_negative"]
        n_total = stats["n_samples"]
        minority_count = min(n_pos, n_neg)

        if min_minority > 0 and minority_count < min_minority:
            continue
        if min_prev > 0 and n_total > 0 and minority_count / n_total < min_prev:
            continue
        if min_subgroup_samples > 0:
            subgroups = stats.get("subgroups", {}).get(subgroup_attr, {})
            if subgroup_attr == "ethnicity":
                # Only check Caucasian and African American
                failed = False
                for label in _ETHNICITY_FOCUS:
                    if int(subgroups.get(label, 0)) < min_subgroup_samples:
                        failed = True
                        break
                if failed:
                    continue
            else:
                if subgroups:
                    min_count = min(int(c) for c in subgroups.values())
                    if min_count < min_subgroup_samples:
                        continue

        # Sort score
        if client_sort_mode == "prevalence":
            score = n_pos / max(n_total, 1)
        else:
            score = n_pos
        selected.append((int(hid), score))

    # Sort descending by score
    selected.sort(key=lambda x: x[1], reverse=True)

    # Top-k selection
    if num_clients > 0 and num_clients < len(selected):
        selected = selected[:num_clients]

    return [hid for hid, _ in selected]


# ---------------------------------------------------------------------------
# Selection directory with symlinks
# ---------------------------------------------------------------------------

def ensure_selection_dir(
    pool_data_dir: Path,
    hospital_ids: List[int],
    sel_hash: str,
    pool_config: dict,
    *,
    outer_kfold: Optional[int] = None,
) -> str:
    """Create a ``selection[{hash}]/`` directory with index-based symlinks.

    Returns the dataset path string (relative, suitable for ``args.dataset``)
    pointing to the selection directory.

    For K-fold data, creates symlinks inside each fold's selection directory.
    For single-split data, creates symlinks directly in the selection directory.
    """
    sel_dir_name = f"selection[{sel_hash}]"

    if outer_kfold is not None:
        # K-fold: create selection dir inside each fold
        for fold_idx in range(outer_kfold):
            fold_dir = pool_data_dir / f"fold_{fold_idx}"
            _create_fold_symlinks(fold_dir, sel_dir_name, hospital_ids)
            _write_selection_config(
                fold_dir / sel_dir_name / "config.json",
                pool_config, hospital_ids, fold_idx=fold_idx,
            )
        # Return path to first fold (caller will handle fold iteration)
        # Actually, the dataset path should NOT include fold — that's added
        # by run_experiments.  Return the base partition path.
        return sel_dir_name
    else:
        # Single split: create selection dir at pool root
        _create_single_split_symlinks(pool_data_dir, sel_dir_name, hospital_ids)
        _write_selection_config(
            pool_data_dir / sel_dir_name / "config.json",
            pool_config, hospital_ids,
        )
        return sel_dir_name


def _ensure_symlink(link: Path, target: Path) -> None:
    """Create or update a symlink, replacing if target differs.

    Handles race conditions from concurrent jobs creating the same symlink.
    """
    if link.is_symlink():
        if link.readlink() == target:
            return
        try:
            link.unlink()
        except FileNotFoundError:
            pass  # another process removed it
    elif link.exists():
        try:
            link.unlink()
        except FileNotFoundError:
            pass
    try:
        link.symlink_to(target)
    except FileExistsError:
        # Another process created it concurrently — verify it's correct
        if link.is_symlink() and link.readlink() == target:
            return
        # Wrong target — retry once
        try:
            link.unlink()
            link.symlink_to(target)
        except (FileExistsError, FileNotFoundError):
            pass  # concurrent race, accept whatever exists


def _create_fold_symlinks(
    fold_dir: Path, sel_dir_name: str, hospital_ids: List[int],
) -> None:
    """Create train/test/demographics symlinks for one fold.

    Directory layout::

        fold_0/
            train/h167.npz          ← actual data
            selection[hash]/
                train/0.npz         ← symlink → ../../train/h167.npz

    Symlinks go up 2 levels: from ``selection[hash]/train/`` to ``fold_0/``.
    """
    sel_dir = fold_dir / sel_dir_name
    for subdir in ("train", "test"):
        target_dir = sel_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for idx, hid in enumerate(hospital_ids):
            link = target_dir / f"{idx}.npz"
            target = Path("..") / ".." / subdir / f"h{hid}.npz"
            _ensure_symlink(link, target)

    demo_dir = sel_dir / "demographics"
    demo_dir.mkdir(parents=True, exist_ok=True)
    for idx, hid in enumerate(hospital_ids):
        for split in ("train", "test"):
            link = demo_dir / f"{idx}_{split}.npz"
            target = Path("..") / ".." / "demographics" / f"h{hid}_{split}.npz"
            _ensure_symlink(link, target)


def _create_single_split_symlinks(
    pool_dir: Path, sel_dir_name: str, hospital_ids: List[int],
) -> None:
    """Create train/test/demographics symlinks for single-split data."""
    sel_dir = pool_dir / sel_dir_name
    for subdir in ("train", "test"):
        target_dir = sel_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for idx, hid in enumerate(hospital_ids):
            link = target_dir / f"{idx}.npz"
            target = Path("..") / ".." / subdir / f"h{hid}.npz"
            _ensure_symlink(link, target)

    demo_dir = sel_dir / "demographics"
    demo_dir.mkdir(parents=True, exist_ok=True)
    for idx, hid in enumerate(hospital_ids):
        for split in ("train", "test"):
            link = demo_dir / f"{idx}_{split}.npz"
            target = Path("..") / ".." / "demographics" / f"h{hid}_{split}.npz"
            _ensure_symlink(link, target)


def _write_selection_config(
    config_path: Path,
    pool_config: dict,
    hospital_ids: List[int],
    fold_idx: Optional[int] = None,
) -> None:
    """Write a config.json for the selection directory.

    Includes client_label_counts in positional order so that
    ``load_client_label_counts()`` works unchanged.
    """
    metadata = pool_config["hospital_metadata"]

    # Build client_label_counts in positional order
    client_label_counts = []
    for hid in hospital_ids:
        stats = metadata[str(hid)]
        client_label_counts.append([
            [0, stats["n_negative"]],
            [1, stats["n_positive"]],
        ])

    sel_config = {
        "mode": "selection",
        "pool_partition_id": pool_config.get("partition_id", ""),
        "num_clients": len(hospital_ids),
        "num_classes": pool_config.get("num_classes", 2),
        "client_ids": hospital_ids,
        "hospital_id_map": {i: hid for i, hid in enumerate(hospital_ids)},
        "client_label_counts": client_label_counts,
        "data_format": pool_config.get("data_format", "ts_static"),
        "max_seq_len": pool_config.get("max_seq_len"),
        "n_ts_features": pool_config.get("n_ts_features"),
        "n_static_features": pool_config.get("n_static_features"),
        "n_flat_features": pool_config.get("n_flat_features"),
        "n_diag_features": pool_config.get("n_diag_features"),
    }
    if fold_idx is not None:
        sel_config["fold_idx"] = fold_idx
        sel_config["outer_kfold"] = pool_config.get("outer_kfold")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(sel_config, f, indent=2)
        f.write("\n")

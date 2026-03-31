"""Lightweight partition-ID utilities (stdlib only — no torch/sklearn).

Canonical implementations live in ``utils.fingerprinting``.
This module re-exports them so that existing import paths continue to work::

    from utils.partition_utils import build_cifar_partition_id  # still works
"""

from utils.fingerprinting import (  # noqa: F401
    # Hash computation
    partition_fingerprint as partition_cfg_hash,
    # Directory name construction
    build_partition_dir_name,
    build_cifar_default_label,
    # High-level partition-ID builders
    build_cifar_partition_id,
    build_eicu_partition_id,
    build_eicu_pool_partition_id,
    # Pool-mode fingerprinting
    model_checkpoint_hash,
    extract_training_hparams,
    pool_level_fingerprint,
    selection_fingerprint,
    # Partition lookup / extraction
    find_by_hash as find_partition_by_hash,
    extract_cfg_hash,
    # Legacy builders (kept for migration scripts only)
    _legacy_dataset_partition_id,
    _legacy_eicu_partition_id,
)

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from flcore.servers.serverbase import Server
from flcore.clients.clientpae import clientPAE
from ensemble.helpers import derive_config_ids, load_base_clf, run_stage
from ensemble.pool_utils import setup_pool_mode, slice_graph_bundle
from ensemble.graph_utils import require_feats, save_feats, _bundle_filename
from ensemble.base_clf_utils import pool_composition_hash
from ensemble.pool_quality import compute_pool_quality_scores

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class FedPAE(Server):


    def __init__(self, args, times):
        super().__init__(args, times)

        # Pool mode: reuse FedDES pool-mode setup
        if getattr(args, "pool_mode", False):
            setup_pool_mode(self, args)

        # Populate slow-client masks so set_clients builds all clients.
        self.set_slow_clients()
        self.set_clients(clientPAE)

        # Convenience names used for clf_key tuples.
        self.models = [f"model_{model_id}" for model_id in range(len(self.args.models))]

    # -------------------------
    # Main training workflow
    # -------------------------
    def _init_classifier_keys(self) -> None:
        """Initialize local and global classifier keys for all clients."""
        global_clf_keys: List[Tuple[str, str]] = []
        for client in self.clients:
            client.local_clf_keys = []
            for model_str in client.model_strs:
                clf_key = (client.role, model_str)
                global_clf_keys.append(clf_key)
                client.local_clf_keys.append(clf_key)

        self.global_clf_keys = global_clf_keys
        pool_sfx = pool_composition_hash(global_clf_keys)
        for client in self.clients:
            client.global_clf_keys = global_clf_keys
            client._pool_suffix = pool_sfx

    def train(self):
        # Build global/local classifier keys once (used by clients and for classifier_pool).
        self._init_classifier_keys()

        phase = str(self.args.phase).lower()

        # -----------------------------
        # Phase 1: Base + DS bundle
        # -----------------------------
        if phase == "1":
            clients_needing_base = [c for c in self.clients if not c.base_classifiers_exist()]
            if clients_needing_base:
                print("[FedPAE][Server] Phase 1.1 starting: train_base_classifiers")
                run_stage(clients_needing_base, stage="train_base_classifiers")

            # Load all base classifiers into a shared pool for DS/meta-label projection.
            role_to_client = {c.role: c for c in self.clients}
            classifier_pool: Dict[Tuple[str, str], Any] = {}

            for client_role, model_str in self.global_clf_keys:
                client = role_to_client[client_role]
                classifier_pool[(client_role, model_str)] = load_base_clf(client, model_str)

            clients_needing_graph_prep = [c for c in self.clients if not c.graph_bundle_exists()]
            if clients_needing_graph_prep:
                print("[FedPAE][Server] Phase 1.2 starting: prepare_graph_data")
                run_stage(
                    clients_needing_graph_prep,
                    stage="prepare_graph_data",
                    stage_inputs={"classifier_pool": classifier_pool},
                )
            else:
                print("[FedPAE][Server] All clients already have decision-space bundles.")
            return None

        # -----------------------------
        # Phase 2: NSGA-II ensemble search
        # -----------------------------
        elif phase == "2":
            self._prune_classifier_pool()

            base_id, pae_id = derive_config_ids(self.args, prefixes=("base", "pae"))
            pae_alias = f"base[{base_id}]_pae[{pae_id}]"
            csv_path = Path(self.args.ckpt_root) / "pae" / pae_alias / "results.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            skip_pae_training = getattr(self.args, "skip_pae_training", False)
            if skip_pae_training and csv_path.exists():
                print(f"[FedPAE][Server] Skipping PAE training because results exist at {csv_path}")
            else:
                print("[FedPAE][Server] Phase 2 starting: run_ensemble_selection")
                run_stage(self.clients, stage="run_ensemble_selection")
                self._log_results(csv_path)

            return None

        elif phase in ("auto", "all"):
            # ---- Phase 1: Base + DS bundle ----
            clients_needing_base = [c for c in self.clients if not c.base_classifiers_exist()]
            if clients_needing_base:
                print("[FedPAE][Server] Phase 1.1 starting: train_base_classifiers")
                run_stage(clients_needing_base, stage="train_base_classifiers")

            role_to_client = {c.role: c for c in self.clients}
            classifier_pool: Dict[Tuple[str, str], Any] = {}
            for client_role, model_str in self.global_clf_keys:
                client = role_to_client[client_role]
                classifier_pool[(client_role, model_str)] = load_base_clf(client, model_str)

            clients_needing_graph_prep = [c for c in self.clients if not c.graph_bundle_exists()]
            if clients_needing_graph_prep:
                print("[FedPAE][Server] Phase 1.2 starting: prepare_graph_data")
                run_stage(
                    clients_needing_graph_prep,
                    stage="prepare_graph_data",
                    stage_inputs={"classifier_pool": classifier_pool},
                )

            # ---- Phase 2: NSGA-II ensemble search ----
            self._prune_classifier_pool()

            base_id, pae_id = derive_config_ids(self.args, prefixes=("base", "pae"))
            pae_alias = f"base[{base_id}]_pae[{pae_id}]"
            csv_path = Path(self.args.ckpt_root) / "pae" / pae_alias / "results.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            skip_pae_training = getattr(self.args, "skip_pae_training", False)
            if skip_pae_training and csv_path.exists():
                print(f"[FedPAE][Server] Skipping PAE training because results exist at {csv_path}")
            else:
                print("[FedPAE][Server] Phase 2 starting: run_ensemble_selection")
                run_stage(self.clients, stage="run_ensemble_selection")
                self._log_results(csv_path)

        else:
            print(f"[FedPAE][Server] Unknown phase {phase} specified. No action taken.")

    # -------------------------
    # Pre-pool pruning
    # -------------------------
    def _prune_classifier_pool(self) -> None:
        """Optionally prune weak classifiers from the global pool before ensemble selection.

        Controlled by ``args.pae_prune_bottom_pct`` (default 0 = no pruning).
        When active: computes cross-client mean minority recall, removes the
        bottom X%, updates ``global_clf_keys`` / ``local_clf_keys``, and saves
        column-sliced graph bundles.  Sets ``client.pruned_bundle_path`` so
        that ``run_ensemble_selection`` loads the pruned version.

        When ``pae_prune_protect_local`` is True, each client's own local
        classifiers are rescued from the prune set (per-client).
        """
        prune_pct = float(getattr(self.args, "pae_prune_bottom_pct", 0))
        if prune_pct <= 0:
            return

        M_orig = len(self.global_clf_keys)
        if M_orig <= 2:
            print("[FedPAE][Server] Pruning skipped: pool has <= 2 classifiers.")
            return

        original_global_clf_keys = list(self.global_clf_keys)

        # 1. Score classifiers using original full-M bundles.
        scores = compute_pool_quality_scores(
            clients=self.clients,
            global_clf_keys=self.global_clf_keys,
            num_classes=self.args.num_classes,
        )
        per_clf = scores.get("per_clf_scores", [])
        if not per_clf:
            print("[FedPAE][Server] Pruning skipped: no quality scores available.")
            return

        # 2. Sort by minority recall (ascending) and determine global prune set.
        sorted_by_recall = sorted(per_clf, key=lambda e: e["mean_minority_recall"])
        n_prune = max(1, int(M_orig * prune_pct / 100.0))
        n_prune = min(n_prune, M_orig - 2)  # keep at least 2 classifiers
        if n_prune <= 0:
            print("[FedPAE][Server] Pruning skipped: would prune 0 classifiers.")
            return

        global_pruned_indices = set()
        for entry in sorted_by_recall[:n_prune]:
            global_pruned_indices.add(entry["clf_index"])

        global_keep_indices = [i for i in range(M_orig) if i not in global_pruned_indices]
        global_keep_keys = [original_global_clf_keys[i] for i in global_keep_indices]
        pruned_keys = [original_global_clf_keys[i] for i in sorted(global_pruned_indices)]
        M_global = len(global_keep_indices)

        protect_local = bool(getattr(self.args, "pae_prune_protect_local", False))

        print(
            f"[FedPAE][Server] Pruning: {len(global_pruned_indices)}/{M_orig} classifiers removed "
            f"(bottom {prune_pct}% by minority recall). "
            f"Global M: {M_orig} -> {M_global}"
            f"{' (per-client locals protected)' if protect_local else ''}"
        )
        for pk in pruned_keys:
            recall = next(
                (e["mean_minority_recall"] for e in per_clf if tuple(e["clf_key"]) == tuple(pk)),
                None,
            )
            print(f"  PRUNED: {pk}  (minority recall={recall:.4f})" if recall else f"  PRUNED: {pk}")

        # 3. Update server-level keys.
        self.global_clf_keys = global_keep_keys

        # 4. Per-client: compute keep set, update keys, slice bundles.
        num_classes = self.args.num_classes
        feat_mode = str(getattr(self.args, "graph_sample_node_feats", "ds"))
        key_to_idx = {tuple(k): i for i, k in enumerate(original_global_clf_keys)}

        base_id, pae_id = derive_config_ids(self.args, prefixes=("base", "pae"))
        pae_alias = f"base[{base_id}]_pae[{pae_id}]"
        pruned_dir = Path(self.args.ckpt_root) / "pae" / pae_alias / "pruned"
        pruned_dir.mkdir(parents=True, exist_ok=True)

        per_client_m_sizes: Dict[str, int] = {}

        for client in self.clients:
            if protect_local:
                local_indices = set()
                for lk in client.local_clf_keys:
                    idx = key_to_idx.get(tuple(lk))
                    if idx is not None:
                        local_indices.add(idx)
                client_keep_indices = sorted(set(global_keep_indices) | local_indices)
                n_rescued = len(local_indices & global_pruned_indices)
                if n_rescued > 0:
                    print(
                        f"  [{client.role}] Rescued {n_rescued} local classifier(s) "
                        f"from prune set (M={len(client_keep_indices)})"
                    )
            else:
                client_keep_indices = global_keep_indices

            client_keep_keys = [original_global_clf_keys[i] for i in client_keep_indices]
            keep_set = set(tuple(k) for k in client_keep_keys)

            client.global_clf_keys = client_keep_keys
            client.local_clf_keys = [
                k for k in client.local_clf_keys if tuple(k) in keep_set
            ]

            per_client_m_sizes[client.role] = len(client_keep_indices)

            # Slice graph bundle + feats and save to pruned directory.
            bundle_path = Path(client.base_dir) / _bundle_filename(client.role, client.pool_suffix)
            if not bundle_path.exists():
                print(f"[FedPAE][Server][warn] No bundle for {client.role}, skipping slice.")
                continue

            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            feats_data = require_feats(Path(client.base_dir), client.role, feat_mode, pool_suffix=client.pool_suffix)
            sliced, sliced_feats = slice_graph_bundle(
                bundle, client_keep_indices, M_orig, num_classes, feat_mode,
                feats_by_split=feats_data,
            )

            out_path = pruned_dir / f"{client.role}_graph_bundle.pt"
            torch.save(sliced, out_path)
            save_feats(pruned_dir, client.role, feat_mode, sliced_feats)
            client.pruned_bundle_path = out_path

        # Summary of per-client M sizes.
        unique_ms = sorted(set(per_client_m_sizes.values()))
        if len(unique_ms) == 1:
            print(f"[FedPAE][Server] Saved {len(self.clients)} pruned bundles (M={unique_ms[0]}).")
        else:
            print(
                f"[FedPAE][Server] Saved {len(self.clients)} pruned bundles. "
                f"Per-client M range: {min(unique_ms)}-{max(unique_ms)} "
                f"(unique sizes: {unique_ms})"
            )

        # 5. Save pruning report.
        pruning_report: Dict[str, Any] = {
            "prune_bottom_pct": prune_pct,
            "protect_local": protect_local,
            "M_original": M_orig,
            "M_global_after_pruning": M_global,
            "n_pruned": len(global_pruned_indices),
            "pruned_classifiers": [
                {
                    "key": list(pruned_keys[i]),
                    "original_index": sorted(global_pruned_indices)[i],
                    "minority_recall": next(
                        (e["mean_minority_recall"] for e in per_clf
                         if tuple(e["clf_key"]) == tuple(pruned_keys[i])),
                        None,
                    ),
                }
                for i in range(len(pruned_keys))
            ],
            "kept_classifiers_global": [list(k) for k in global_keep_keys],
        }
        if protect_local:
            pruning_report["per_client_M"] = per_client_m_sizes

        report_path = pruned_dir / "pruning_report.json"
        with open(report_path, "w") as f:
            json.dump(pruning_report, f, indent=2)
        print(f"[FedPAE][Server] Pruning report saved to {report_path}")

    def _log_results(self, csv_path: Path) -> None:
        client_summaries = [(client.role, getattr(client, "perf_summary", {})) for client in self.clients]

        aggregate: Dict[str, Any] = {"n_clients": len(client_summaries)}
        mean_keys = ["local_acc", "local_bacc", "global_acc", "global_bacc", "FedPAE_acc", "FedPAE_bacc"]
        sum_keys = [
            "acc_beats_local",
            "bacc_beats_local",
            "acc_beats_global",
            "bacc_beats_global",
            "acc_beats_baselines",
            "bacc_beats_baselines",
            "acc_ties_local",
            "bacc_ties_local",
            "acc_ties_global",
            "bacc_ties_global",
            "acc_ties_baselines",
            "bacc_ties_baselines",
        ]

        for key in mean_keys:
            vals = [s.get(key) for _, s in client_summaries]
            aggregate[f"mean_{key}"] = float(sum(vals) / len(vals)) if vals else float("nan")

        for key in sum_keys:
            vals = [s.get(key) for _, s in client_summaries]
            aggregate[f"{key}"] = int(sum(vals))

        excluded_table_cols = {"acc_beats_baselines", "bacc_beats_baselines", "acc_ties_baselines", "bacc_ties_baselines"}
        all_metric_keys = {k for _, s in client_summaries for k in s}
        metric_cols = sorted(all_metric_keys - excluded_table_cols)
        table_cols = ["client"] + metric_cols

        with csv_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(table_cols)
            for role, summary in client_summaries:
                writer.writerow([role] + [summary.get(col) for col in metric_cols])

        self.perf_summary = aggregate

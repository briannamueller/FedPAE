# from __future__ import annotations
# # system/flcore/clients/clientdes.py
# import json
# import random
# from types import SimpleNamespace

# import numpy as np
# import torch
# from sklearn.metrics import balanced_accuracy_score
# from pathlib import Path
# from types import SimpleNamespace
# from typing import Any, Dict, List, Optional
# import os
# import shutil
# import torch
# import torch.nn.functional as F
# import numpy as np
# import time
# import wandb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F  # local import to support one-hot voting
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, f1_score
# import random
# from types import SimpleNamespace

# import numpy as np
# import torch
# from sklearn.metrics import balanced_accuracy_score
# # Shared training helpers

# # ---------- DES utilities ----------
# from des.graph_utils import build_train_eval_graph, project_to_DS
# from des.base_clf_utils import fit_clf
# from des.helpers import derive_config_ids, init_base_meta_loaders, get_performance_baselines

# from flcore.clients.clientbase import Client
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from deap import base, creator, tools, algorithms
from sklearn.metrics import balanced_accuracy_score

from ensemble.graph_utils import project_to_DS, save_feats, _bundle_filename, _feats_filename
from ensemble.base_clf_utils import fit_clf, fit_tree_clf, is_tree_model, seed_for_model
from ensemble.helpers import derive_config_ids, init_base_meta_loaders
from flcore.clients.clientbase import Client
from utils.fingerprinting import model_checkpoint_hash, extract_training_hparams, pool_level_fingerprint

class clientPAE(Client):

    def __init__(self, args, id: int, train_samples: int, test_samples: int, **kwargs: Any) -> None:
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.args = args
        self.device = args.device
        total_models = len(args.models)
        if total_models == 0:
            raise ValueError("FedPAE expects at least one model in args.models.")
        self.base_single_model = getattr(args, "base_single_model", False)
        if self.base_single_model:
            model_ids = [self.id % total_models]
        else:
            model_ids = list(range(total_models))
        self.model_ids = model_ids
        self.model_strs = [f"model_{model_id}" for model_id in model_ids]
        self.num_models = len(model_ids)

        self._pool_mode = getattr(args, "pool_mode", False)
        if self._pool_mode and hasattr(args, "hospital_ids"):
            self.hospital_id = args.hospital_ids[id]
        else:
            self.hospital_id = id

        if self._pool_mode:
            training_hparams = extract_training_hparams(args)
            self.model_dirs: Dict[int, Path] = {}
            model_hashes = []
            for mid in self.model_ids:
                mhash = model_checkpoint_hash(args.models[mid], training_hparams)
                self.model_dirs[mid] = args.ckpt_root / "base_clf" / f"model[{mhash}]"
                model_hashes.append(mhash)
            hospital_ids = list(getattr(args, "hospital_ids", []))
            pool_fp = pool_level_fingerprint(model_hashes, hospital_ids=hospital_ids)
            _, self.pae_config_id = derive_config_ids(self.args, prefixes=("base", "pae"))
            self.base_config_id = pool_fp
            self.base_dir = args.ckpt_root / "base_clf" / f"pool[{pool_fp}]"
            self.base_outputs_dir = args.outputs_root / "base_clf" / f"pool[{pool_fp}]"
            for d in self.model_dirs.values():
                d.mkdir(parents=True, exist_ok=True)
        else:
            self.base_config_id, self.pae_config_id = derive_config_ids(self.args, prefixes=("base", "pae"))
            self.base_dir = args.ckpt_root / "base_clf" /  f"base[{self.base_config_id}]"
            self.base_outputs_dir = args.outputs_root / "base_clf" /  f"base[{self.base_config_id}]"
            self.model_dirs = {}

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_outputs_dir.mkdir(parents=True, exist_ok=True)

        self._base_train_loader = None
        self._meta_train_loader = None
        self.meta_history: List[Dict[str, Any]] = []
        self._pool_suffix = ""

    @property
    def pool_suffix(self) -> str:
        return self._pool_suffix

    def _ckpt_path(self, model_id: int, model_str: str) -> Path:
        model_def = self.args.models[model_id]
        ext = "pkl" if is_tree_model(model_def) else "pt"
        if self._pool_mode and self.model_dirs:
            return self.model_dirs[model_id] / f"h{self.hospital_id}.{ext}"
        return self.base_dir / f"{self.role}_{model_str}.{ext}"

    def base_classifiers_exist(self) -> bool:
        expected = [
            self._ckpt_path(model_id, model_str)
            for model_id, model_str in zip(self.model_ids, self.model_strs)
        ]
        statuses = {str(p): p.exists() for p in expected}
        ok = all(statuses.values()) if expected else True
        return ok

    def graph_bundle_exists(self) -> bool:
        bundle_path = Path(self.base_dir) / _bundle_filename(self.role, self.pool_suffix)
        if not bundle_path.exists():
            return False
        feat_mode = str(getattr(self.args, "graph_sample_node_feats", "ds")).lower()
        feats_path = Path(self.base_dir) / _feats_filename(self.role, feat_mode, self.pool_suffix)
        return feats_path.exists()

    def train_base_classifiers(self, device=None):
        device = torch.device(device if device is not None else self.device)
        base_train_loader, _ = init_base_meta_loaders(self)
        val_loader = self.load_val_data()

        for model_id, model_str in zip(self.model_ids, self.model_strs):
            ckpt_path = self._ckpt_path(model_id, model_str)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            if ckpt_path.exists():
                print(f"[FedPAE][Client {self.role}] Skipping {model_str}: checkpoint exists")
                continue
            model_def = self.args.models[model_id]
            seed_for_model(self.id, model_id, hospital_id=self.hospital_id if self._pool_mode else None)
            if is_tree_model(model_def):
                best_epoch, best_score, model = fit_tree_clf(
                    self, model_id, model_def,
                    base_train_loader, val_loader,
                )
                import joblib
                joblib.dump(model, ckpt_path)
            else:
                best_epoch, best_score, model = fit_clf(
                    self,
                    model_id, base_train_loader, val_loader, device,
                    max_epochs=self.args.local_epochs,
                    patience=self.args.base_es_patience,
                    es_metric=self.args.base_es_metric,
                    lr=self.args.base_clf_lr,
                    min_delta=self.args.base_es_min_delta,
                )
                torch.save(model.cpu(), ckpt_path)
            print(f"{self.role} {model_id} stopping training at epoch {best_epoch}, score = {best_score}")
    

    def prepare_graph_data(self, device=None, classifier_pool: Dict[Any, torch.nn.Module] = None) -> None:
        """
        Prepare decision-space and meta-data for graph building.

        Steps:
        1) Evaluate classifier pool → decision-space (DS), preds, labels,
            meta-labels, and sample features for train/val/test.
        2) Optionally load/save these tensors from/to a flat cache.
        """

        device = torch.device(device if device is not None else self.device)
        self.device = device

        _, meta_train_loader = init_base_meta_loaders(self)
        val_loader = self.load_val_data()
        test_loader = self.load_test_data()

        data_loaders = {"train": meta_train_loader, "val": val_loader, "test": test_loader}

        # Flat cache for graph inputs (bundle) + separate feats cache per mode.
        bundle_path = Path(self.base_dir) / _bundle_filename(self.role, self.pool_suffix)
        feat_mode = str(getattr(self.args, "graph_sample_node_feats", "ds")).lower()
        feats_path = Path(self.base_dir) / _feats_filename(self.role, feat_mode, self.pool_suffix)

        bundle_exists = bundle_path.exists()
        feats_exist = feats_path.exists()

        if bundle_exists and feats_exist:
            print(f"[FedPAE][Client {self.role}] Cached bundle + feats ({feat_mode}) found.")
            return

        # Need forward passes (for bundle, feats, or both).
        print(f"[FedPAE][Client {self.role}] Computing {'bundle + ' if not bundle_exists else ''}feats ({feat_mode})...")
        graph_data = {}
        feats_data = {}
        for data_split in ["train", "val", "test"]:
            loader = data_loaders[data_split]
            ds, preds, y_true, meta_labels, feats = project_to_DS(
                self, loader, classifier_pool
            )
            graph_data[data_split] = {
                "ds": ds, "preds": preds, "y": y_true, "meta": meta_labels.float()
            }
            feats_data[data_split] = feats

        if not bundle_exists:
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph_data, bundle_path)
        save_feats(self.base_dir, self.role, feat_mode, feats_data, pool_suffix=self.pool_suffix)

    def _compute_ensemble_metrics(
        self,
        ds: torch.Tensor,
        y_true: torch.Tensor,
        selected_indices: list[int],
    ) -> dict[str, float]:
        """Compute accuracy and balanced accuracy for an ensemble.

        Args:
            ds: Decision-space predictions of shape [N, M * C], where
                N is #samples, M is #classifiers, C is #classes.
                Entries are per-class probabilities for each classifier.
            y_true: Ground-truth labels of shape [N].
            selected_indices: Indices of classifiers to include in the ensemble.

        Returns:
            A dict with keys "acc" and "bacc".
        """
        if len(selected_indices) == 0:
            return {"acc": 0.0, "bacc": 0.0}

        num_classes = self.args.num_classes
        combination_mode = getattr(self.args, "pae_combination_mode", "soft")
        N = ds.size(0)
        M = ds.size(1) // num_classes

        # [N, M, C]
        ds_reshaped = ds.view(N, M, num_classes)
        selected_ds = ds_reshaped[:, selected_indices, :]

        if combination_mode == "hard":
            # Majority vote on per-classifier argmax predictions
            per_clf_preds = selected_ds.argmax(dim=2)  # [N, K]
            # One-hot vote tallies: [N, K, C] summed over K -> [N, C]
            votes = torch.zeros(N, num_classes, device=ds.device)
            for k in range(len(selected_indices)):
                votes.scatter_add_(1, per_clf_preds[:, k:k+1], torch.ones(N, 1, device=ds.device))
            preds = votes.argmax(dim=1)
        else:
            # Soft: average probabilities, then argmax
            avg_probs = selected_ds.mean(dim=1)
            preds = avg_probs.argmax(dim=1)

        preds_np = preds.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()

        acc = (preds == y_true).float().mean().item()

        try:
            bacc = balanced_accuracy_score(y_true_np, preds_np)
        except ValueError:
            # e.g., only one class present in y_true
            bacc = float("nan")

        return {"acc": acc, "bacc": bacc}
    

    def run_ensemble_selection(self, device=None) -> None:
        """Run the FedPAE ensemble selection stage on this client.

        Steps (per FedPAE paper, client-side):
          1. Load cached decision-space predictions for val/test.
          2. Treat the global model bench as the candidate set.
          3. Use NSGA-II to optimize:
               - strength: average individual val accuracy of selected models,
               - diversity: average pairwise independence of their probability
                 outputs (1 - cosine similarity).
          4. From the Pareto set, choose the ensemble with highest ensemble
             val accuracy (using probability averaging for the ensemble).
          5. Evaluate this ensemble on test and compare to:
               - local-only ensemble,
               - global bench ensemble.
        """
        # Device
        device = torch.device(device if device is not None else self.device)
        self.device = device

        # Load DS bundle — prefer pruned bundle if server set one
        bundle_path = getattr(self, "pruned_bundle_path", None)
        if bundle_path is None or not Path(bundle_path).exists():
            bundle_path = self.base_dir / _bundle_filename(self.role, self.pool_suffix)
        if not Path(bundle_path).exists():
            print(f"[FedPAE][Client {self.role}] No decision-space bundle at {bundle_path}; skipping ensemble selection.")
            return

        graph_data = torch.load(bundle_path, map_location="cpu")

        # Only val and test needed
        val = SimpleNamespace(**graph_data["val"])
        test = SimpleNamespace(**graph_data["test"])

        val.ds = val.ds.to(device)
        val.y = val.y.to(device)
        test.ds = test.ds.to(device)
        test.y = test.y.to(device)

        # If no validation data, fallback to baselines only
        if val.ds.size(0) == 0:
            print(f"[FedPAE][Client {self.role}] Empty validation set; computing baselines only.")
            num_classes = self.args.num_classes
            M = test.ds.size(1) // num_classes
            global_indices = list(range(M))

            M_global = len(getattr(self, "global_clf_keys", []))
            if M_global == M:
                local_indices = [idx for idx, (role, _) in enumerate(self.global_clf_keys) if role == self.role]
            else:
                local_indices = global_indices

            local_metrics = self._compute_ensemble_metrics(test.ds, test.y, local_indices)
            global_metrics = self._compute_ensemble_metrics(test.ds, test.y, global_indices)

            FedPAE_acc = 0.0
            FedPAE_bacc = 0.0

            self.perf_summary = {
                "local_acc": local_metrics["acc"],
                "local_bacc": local_metrics["bacc"],
                "global_acc": global_metrics["acc"],
                "global_bacc": global_metrics["bacc"],
                "FedPAE_acc": FedPAE_acc,
                "FedPAE_bacc": FedPAE_bacc,
                "acc_beats_local": 0,
                "bacc_beats_local": 0,
                "acc_beats_global": 0,
                "bacc_beats_global": 0,
                "acc_beats_baselines": 0,
                "bacc_beats_baselines": 0,
                "acc_ties_local": 0,
                "bacc_ties_local": 0,
                "acc_ties_global": 0,
                "bacc_ties_global": 0,
                "acc_ties_baselines": 0,
                "bacc_ties_baselines": 0,
            }
            self.pareto_front = {}
            return

        # -------------------------
        # Precompute per-model stats on val set
        # -------------------------
        num_classes = self.args.num_classes
        N_val = val.ds.size(0)
        M = val.ds.size(1) // num_classes

        # Identify local vs global indices using global_clf_keys
        M_global = len(getattr(self, "global_clf_keys", []))
        if M_global == M:
            local_indices = [idx for idx, (role, _) in enumerate(self.global_clf_keys) if role == self.role]
        else:
            local_indices = list(range(M))

        val_ds_reshaped = val.ds.view(N_val, M, num_classes)
        val_probs = val_ds_reshaped.detach().cpu().numpy()  # [N_val, M, C]
        y_val_np = val.y.detach().cpu().numpy()

        # Eval metric used for individual classifier scoring and ensemble selection
        eval_metric = str(getattr(self.args, "pae_eval_metric", "acc")).lower()

        # Individual scores (strength objective)
        indiv_acc = np.zeros(M, dtype=np.float64)
        for m in range(M):
            preds_m = val_probs[:, m, :].argmax(axis=1)
            if eval_metric == "bacc":
                indiv_acc[m] = balanced_accuracy_score(y_val_np, preds_m)
            else:
                indiv_acc[m] = (preds_m == y_val_np).mean()

        # Diversity precomputation
        diversity_measure = str(getattr(self.args, "pae_diversity_measure", "pang")).lower()

        div_matrix = None       # used by pairwise measures ("cosine", "double_fault")
        nonmax_normed = None    # used by "pang"

        if diversity_measure == "pang":
            # Pang et al. 2019: per-sample L2-normalized non-maximal prediction vectors
            L = num_classes
            nonmax_normed = np.zeros((N_val, L - 1, M), dtype=np.float32)
            for i in range(N_val):
                yi = int(y_val_np[i])
                for m in range(M):
                    probs_m = val_probs[i, m, :]
                    nm = np.concatenate([probs_m[:yi], probs_m[yi + 1:]])
                    norm = np.linalg.norm(nm) + 1e-12
                    nonmax_normed[i, :, m] = nm / norm

        elif diversity_measure == "double_fault":
            preds_all = val_probs.argmax(axis=2)  # [N_val, M]
            wrong = preds_all != y_val_np[:, None]  # [N_val, M]
            div_matrix = np.zeros((M, M), dtype=np.float32)
            for i in range(M):
                for j in range(i + 1, M):
                    both_wrong = float((wrong[:, i] & wrong[:, j]).mean())
                    div_matrix[i, j] = 1.0 - both_wrong
                    div_matrix[j, i] = 1.0 - both_wrong

        else:  # cosine
            flat_probs = []
            for m in range(M):
                vec = val_probs[:, m, :].reshape(-1).astype(np.float32)
                vec = vec - vec.mean()
                norm = np.linalg.norm(vec) + 1e-12
                flat_probs.append(vec / norm)
            div_matrix = np.zeros((M, M), dtype=np.float32)
            for i in range(M):
                for j in range(i + 1, M):
                    independence = 1.0 - float(np.dot(flat_probs[i], flat_probs[j]))
                    div_matrix[i, j] = independence
                    div_matrix[j, i] = independence

        # -------------------------
        # NSGA-II via DEAP
        # -------------------------
        pop_size = int(getattr(self.args, "pae_pop_size", 40))
        num_generations = int(getattr(self.args, "pae_num_generations", 40))
        mutation_prob = float(getattr(self.args, "pae_mutation_prob", 0.05))
        crossover_prob = float(getattr(self.args, "pae_crossover_prob", 0.9))
        lambda_multiple = float(getattr(self.args, "pae_lambda_multiple", 2.0))

        # Determine which ensemble sizes to search over
        sizes_str = getattr(self.args, "pae_ensemble_sizes", None) or ""
        fixed_k = getattr(self.args, "pae_ensemble_size", None)

        if sizes_str:
            # Multi-size sweep: run NSGA-II once per k, pick best across all
            k_values = [max(1, min(int(s.strip()), M)) for s in sizes_str.split(",") if s.strip()]
        elif fixed_k is not None:
            k_values = [max(1, min(int(fixed_k), M))]
        else:
            # Single run with variable size (original behavior)
            min_ens = int(getattr(self.args, "pae_min_ensemble_size", 1))
            raw_max = int(getattr(self.args, "pae_max_ensemble_size", 0))
            max_ens = M if raw_max <= 0 else raw_max
            min_ens = max(1, min(min_ens, M))
            max_ens = max(min_ens, min(max_ens, M))
            if diversity_measure == "pang":
                max_ens = max(min_ens, min(max_ens, num_classes - 1))
            k_values = [None]  # None signals variable-size mode
            _var_min, _var_max = min_ens, max_ens

        # -- DEAP creator types (once) --
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        def _run_nsga(k_fixed):
            """Run one NSGA-II search. k_fixed=int for fixed size, None for variable."""
            if k_fixed is not None:
                min_k, max_k = k_fixed, k_fixed
            else:
                min_k, max_k = _var_min, _var_max

            # -- Objective function --
            def evaluate(mask):
                selected = [i for i, v in enumerate(mask) if v == 1]
                if not selected:
                    return 0.0, 0.0
                sel = np.array(selected)
                strength = float(indiv_acc[sel].mean())
                if sel.size < 2:
                    diversity = 0.0
                elif diversity_measure == "pang":
                    if sel.size > num_classes - 1:
                        diversity = 0.0
                    else:
                        M_sel = nonmax_normed[:, :, sel]
                        gram = np.einsum("nlk,nlj->nkj", M_sel, M_sel)
                        dets = np.linalg.det(gram)
                        diversity = float(np.clip(dets, 0.0, None).mean())
                else:
                    sub = div_matrix[np.ix_(sel, sel)]
                    iu, ju = np.triu_indices(sub.shape[0], k=1)
                    diversity = float(sub[iu, ju].mean()) if iu.size > 0 else 0.0
                return strength, diversity

            def repair(mask):
                ones = [i for i, v in enumerate(mask) if v == 1]
                zeros = [i for i, v in enumerate(mask) if v == 0]
                while len(ones) < min_k and zeros:
                    pick = random.choice(zeros)
                    zeros.remove(pick)
                    ones.append(pick)
                    mask[pick] = 1
                while len(ones) > max_k:
                    drop = random.choice(ones)
                    ones.remove(drop)
                    mask[drop] = 0
                return mask

            def create_population(npop):
                population = []
                while len(population) < npop:
                    k = random.randint(min_k, max_k)
                    chosen = random.sample(range(M), k)
                    mask = [0] * M
                    for idx in chosen:
                        mask[idx] = 1
                    population.append(creator.Individual(mask))
                return population

            def uniform_crossover(ind1, ind2):
                for i in range(len(ind1)):
                    if random.random() < 0.5:
                        ind1[i], ind2[i] = ind2[i], ind1[i]
                repair(ind1)
                repair(ind2)
                return ind1, ind2

            def mutate(mask):
                flip_prob = 1.0 / M
                for i in range(len(mask)):
                    if random.random() < flip_prob:
                        mask[i] = 1 - mask[i]
                repair(mask)
                return mask,

            # Build toolbox
            toolbox = base.Toolbox()
            init_pop = create_population(pop_size)
            pop_iter = iter(init_pop)
            toolbox.register("individual", lambda: next(pop_iter))
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", uniform_crossover)
            toolbox.register("mutate", mutate)
            toolbox.register("select", tools.selNSGA2)

            population = toolbox.population(n=pop_size)

            # Seed slot 0 with local classifiers
            if local_indices:
                local_mask = [0] * M
                for idx in local_indices:
                    local_mask[idx] = 1
                population[0] = creator.Individual(repair(local_mask))

            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            lambda_size = int(lambda_multiple * pop_size)
            algorithms.eaMuPlusLambda(
                population, toolbox,
                mu=pop_size, lambda_=lambda_size,
                cxpb=crossover_prob, mutpb=mutation_prob,
                ngen=num_generations,
                stats=None, halloffame=None, verbose=False,
            )

            return population

        # -- Run NSGA-II for each k value --
        overall_best_acc = -1.0
        overall_best_indices = None
        overall_best_size = None
        overall_best_div = -1.0
        overall_pareto_front = {}

        for k_val in k_values:
            label = f"k={k_val}" if k_val is not None else "variable"
            print(f"[FedPAE][Client {self.role}] Running NSGA-II ({label}, M={M})")

            population = _run_nsga(k_val)

            # Extract Pareto front
            final_fitnesses = [ind.fitness.values for ind in population]
            fronts = tools.sortNondominated(population, len(population))
            pareto_front_inds = fronts[0] if fronts else []

            if not pareto_front_inds:
                print(f"[FedPAE][Client {self.role}]   {label}: empty Pareto front, skipping")
                continue

            pop_index = {id(ind): i for i, ind in enumerate(population)}

            for ind in pareto_front_inds:
                selected = [i for i, v in enumerate(ind) if v == 1]
                if not selected:
                    continue
                metrics = self._compute_ensemble_metrics(val.ds, val.y, selected)
                ens_acc = metrics[eval_metric]
                idx = pop_index[id(ind)]
                _, diversity = final_fitnesses[idx]
                size = len(selected)

                if (
                    ens_acc > overall_best_acc
                    or (ens_acc == overall_best_acc and (overall_best_size is None or size < overall_best_size))
                    or (ens_acc == overall_best_acc and size == overall_best_size and diversity > overall_best_div)
                ):
                    overall_best_acc = ens_acc
                    overall_best_size = size
                    overall_best_div = diversity
                    overall_best_indices = selected

                    # Store this run's Pareto front as the displayed one
                    all_fitnesses = list(final_fitnesses)
                    pareto_indices = [pop_index[id(x)] for x in pareto_front_inds if id(x) in pop_index]
                    overall_pareto_front = {
                        "all_fitnesses": all_fitnesses,
                        "pareto_indices": pareto_indices,
                        "selected_strength": ens_acc,
                        "selected_diversity": diversity,
                        "selected_size": size,
                    }

            print(f"[FedPAE][Client {self.role}]   {label}: best val_acc={overall_best_acc:.4f}")

        # Final selection
        if overall_best_indices is None:
            print(f"[FedPAE][Client {self.role}] No valid ensemble found; fallback to local baseline.")
            selected_indices = local_indices if local_indices else list(range(M))
            self.pareto_front = {}
        else:
            selected_indices = overall_best_indices
            self.pareto_front = overall_pareto_front

        print(f"[FedPAE][Client {self.role}] Selected ensemble with {len(selected_indices)} models: {selected_indices}")

        # -------------------------
        # Evaluate on test + baselines
        # -------------------------
        FedPAE_metrics = self._compute_ensemble_metrics(test.ds, test.y, selected_indices)
        FedPAE_acc = FedPAE_metrics["acc"]
        FedPAE_bacc = FedPAE_metrics["bacc"]

        global_indices = list(range(M))
        local_metrics = self._compute_ensemble_metrics(test.ds, test.y, local_indices)
        global_metrics = self._compute_ensemble_metrics(test.ds, test.y, global_indices)

        self.perf_summary = {
            "local_acc": local_metrics["acc"],
            "local_bacc": local_metrics["bacc"],
            "global_acc": global_metrics["acc"],
            "global_bacc": global_metrics["bacc"],
            "FedPAE_acc": FedPAE_acc,
            "FedPAE_bacc": FedPAE_bacc,
            "acc_beats_local": int(FedPAE_acc > local_metrics["acc"]),
            "bacc_beats_local": int(FedPAE_bacc > local_metrics["bacc"]),
            "acc_beats_global": int(FedPAE_acc > global_metrics["acc"]),
            "bacc_beats_global": int(FedPAE_bacc > global_metrics["bacc"]),
            "acc_beats_baselines": int((FedPAE_acc > local_metrics["acc"]) + (FedPAE_acc > global_metrics["acc"])),
            "bacc_beats_baselines": int((FedPAE_bacc > local_metrics["bacc"]) + (FedPAE_bacc > global_metrics["bacc"])),
            "acc_ties_local": int(FedPAE_acc == local_metrics["acc"]),
            "bacc_ties_local": int(FedPAE_bacc == local_metrics["bacc"]),
            "acc_ties_global": int(FedPAE_acc == global_metrics["acc"]),
            "bacc_ties_global": int(FedPAE_bacc == global_metrics["bacc"]),
            "acc_ties_baselines": int((FedPAE_acc == local_metrics["acc"]) + (FedPAE_acc == global_metrics["acc"])),
            "bacc_ties_baselines": int((FedPAE_bacc == local_metrics["bacc"]) + (FedPAE_bacc == global_metrics["bacc"])),
        }

        print(
            f"[FedPAE][Client {self.role}] Test metrics -- "
            f"FedPAE_acc={FedPAE_acc:.4f}, FedPAE_bacc={FedPAE_bacc:.4f}, "
            f"local_acc={local_metrics['acc']:.4f}, global_acc={global_metrics['acc']:.4f}"
        )

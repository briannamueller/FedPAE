"""Argument parser for FedPAE experiments."""

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    parser.add_argument('-go', "--goal", type=str, default="test")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedPAE")
    parser.add_argument('-dev', "--device", type=str, default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument('-did', "--device_id", type=str, default=None)
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="HtFE-img-4")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01)
    parser.add_argument('-ld', "--learning_rate_decay", type=str2bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=1)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1)
    parser.add_argument('-nc', "--num_clients", type=int, default=20)
    parser.add_argument('-pv', "--prev", type=int, default=0)
    parser.add_argument('-t', "--times", type=int, default=1)
    parser.add_argument('-eg', "--eval_gap", type=int, default=1)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-mfn', "--models_folder_name", type=str, default='')
    parser.add_argument('-tc', "--top_cnt", type=int, default=50)
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=True)
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0)
    parser.add_argument('-rjr', "--random_join_ratio", type=str2bool, default=False)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0)
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0)
    parser.add_argument('-ts', "--time_select", type=str2bool, default=False)
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000)

    # -------------------------------------------------------------------------
    # Paths and checkpointing
    # -------------------------------------------------------------------------
    parser.add_argument('--ckpt_root', type=str, default="ckpts")
    parser.add_argument('--outputs_root', type=str, default="../outputs")
    parser.add_argument('--results_dir', type=str, default="../results")
    parser.add_argument('--phase', type=str, default="auto",
                        help="Pipeline phase: 1=base classifiers + DS bundle, "
                             "2=NSGA-II ensemble selection, auto=all.")

    # -------------------------------------------------------------------------
    # Data partition
    # -------------------------------------------------------------------------
    parser.add_argument('--partition-type', type=str, default="dir",
                        choices=["pat", "dir", "exdir"])
    parser.add_argument('--partition-alpha', type=float, default=0.1,
                        help="Dirichlet alpha for non-IID partitioning.")
    parser.add_argument('--partition-C', type=int, default=5)
    parser.add_argument('--partition-min-size', type=int, default=10)
    parser.add_argument('--partition-train-ratio', type=float, default=0.75)
    parser.add_argument('--partition-seed', type=int, default=1)
    parser.add_argument('--data-partition', type=str, default="")
    parser.add_argument('--dataset-hash', type=str, default="")

    # -------------------------------------------------------------------------
    # Pool mode (for pre-partitioned datasets)
    # -------------------------------------------------------------------------
    parser.add_argument('--pool-mode', action='store_true', default=False)
    parser.add_argument('--min-minority', type=int, default=0)
    parser.add_argument('--min-prev', type=float, default=0.0)
    parser.add_argument('--min-subgroup-samples', type=int, default=0)
    parser.add_argument('--subgroup-attr', type=str, default="ethnicity")
    parser.add_argument('--client-sort-mode', type=str, default="prevalence",
                        choices=("positives", "prevalence"))

    # -------------------------------------------------------------------------
    # Validation split and evaluation
    # -------------------------------------------------------------------------
    parser.add_argument('--use_val', type=str2bool, default=True)
    parser.add_argument('--val_ratio', type=float, default=0.25)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--use_bacc_metric', type=str2bool, default=False)
    parser.add_argument('--weighted_loss', type=str2bool, default=False)

    # -------------------------------------------------------------------------
    # Base classifier training (Phase 1)
    # -------------------------------------------------------------------------
    parser.add_argument('--base_single_model', type=str2bool, default=False)
    parser.add_argument('--client_id', type=int, default=None)
    parser.add_argument('--base_split_mode', type=str, default="split_train",
                        choices=["split_train", "oof_stacking"])
    parser.add_argument('--base_split_seed', type=int, default=1)
    parser.add_argument('--base_es_metric', type=str, default="val_loss")
    parser.add_argument('--base_es_patience', type=int, default=20)
    parser.add_argument('--base_es_min_delta', type=float, default=0.0001)
    parser.add_argument('--base_clf_lr', type=float, default=0.0005)
    parser.add_argument('--base_optimizer', type=str, default="Adam",
                        choices=["SGD", "Adam"])
    parser.add_argument('--base_weighted_by_class', type=str2bool, default=True)
    parser.add_argument('--pool_calib_method', type=str, default="ts-mix",
                        choices=["ts-mix", "logistic"])
    parser.add_argument('--graph_sample_node_feats', type=str, default="ds")
    parser.add_argument('--cpu_workers', type=int, default=1)

    # XGBoost / RandomForest
    parser.add_argument('--xgb_n_estimators', type=int, default=100)
    parser.add_argument('--xgb_max_depth', type=int, default=4)
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1)
    parser.add_argument('--xgb_subsample', type=float, default=0.8)
    parser.add_argument('--xgb_colsample_bytree', type=float, default=0.8)
    parser.add_argument('--xgb_min_child_weight', type=int, default=5)
    parser.add_argument('--rf_n_estimators', type=int, default=200)
    parser.add_argument('--rf_max_depth', type=int, default=10)

    # -------------------------------------------------------------------------
    # Graph construction (DS bundle, Phase 1)
    # -------------------------------------------------------------------------
    parser.add_argument('--graph_k_per_class', type=int, default=5)
    parser.add_argument('--graph_cs_topk', type=int, default=3)
    parser.add_argument('--graph_cs_mode', type=str, default="balanced_acc:logloss")
    parser.add_argument('--graph_meta_min_pos', type=int, default=3)
    parser.add_argument('--graph_prune_bottom_pct', type=float, default=0)
    parser.add_argument('--graph_prune_metric', type=str, default="minority_recall",
                        choices=["minority_recall", "bacc", "acc"])
    parser.add_argument('--graph_prune_protect_local', type=str2bool, default=True)

    # -------------------------------------------------------------------------
    # NSGA-II ensemble selection (Phase 2)
    # -------------------------------------------------------------------------
    parser.add_argument('--pae_pop_size', type=int, default=40,
                        help="NSGA-II population size.")
    parser.add_argument('--pae_num_generations', type=int, default=40,
                        help="Number of NSGA-II generations.")
    parser.add_argument('--pae_mutation_prob', type=float, default=0.05,
                        help="Per-individual mutation probability.")
    parser.add_argument('--pae_crossover_prob', type=float, default=0.9,
                        help="Crossover probability.")
    parser.add_argument('--pae_diversity_measure', type=str, default="pang",
                        choices=["pang", "cosine", "double_fault"],
                        help="Diversity objective: pang=DPP-based, cosine=1-cosine sim, "
                             "double_fault=1-P(both wrong).")
    parser.add_argument('--pae_ensemble_size', type=int, default=None,
                        help="Fixed ensemble size. Overrides pae_min/max_ensemble_size.")
    parser.add_argument('--pae_ensemble_sizes', type=str, default="",
                        help="Comma-separated list of sizes to sweep (e.g. '2,3,4'). "
                             "Best across all sizes is selected.")
    parser.add_argument('--pae_min_ensemble_size', type=int, default=1)
    parser.add_argument('--pae_max_ensemble_size', type=int, default=0,
                        help="Maximum ensemble size. 0 = unconstrained (up to M).")
    parser.add_argument('--pae_lambda_multiple', type=float, default=2.0,
                        help="Lambda = pae_lambda_multiple * pop_size for eaMuPlusLambda.")
    parser.add_argument('--pae_combination_mode', type=str, default='soft',
                        choices=['soft', 'hard'],
                        help="soft=average probabilities, hard=majority vote.")
    parser.add_argument('--pae_eval_metric', type=str, default='acc',
                        choices=['acc', 'bacc'],
                        help="Metric used to score individuals and select the best ensemble.")
    parser.add_argument('--pae_prune_bottom_pct', type=float, default=0,
                        help="Remove bottom X%% of classifiers before ensemble selection. 0=off.")
    parser.add_argument('--pae_prune_protect_local', type=str2bool, default=False,
                        help="If True, each client's own classifiers are protected from pruning.")
    parser.add_argument('--skip_pae_training', type=str2bool, default=False,
                        help="Skip Phase 2 if results already exist.")

    return parser

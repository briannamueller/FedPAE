import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def build_ss_edges_cmdw(
    decision_matrix: np.ndarray,
    label_vector: np.ndarray,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
    k_per_class: int = 2,
    membership_mode: str = "soft",   # {"none", "hard", "soft"}
    membership_k: int = 7,
    eps: float = 1e-8,
    log_class_edge_stats: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sample->sample edges using CMDW-style class scores and within-class softmax.

    For each destination j and each class c among SOURCE samples:
      1) Select up to k_per_class nearest class-c sources to j (L1 in decision space), excluding j itself if same class.
      2) CMDW score s_c = m_c / (d_c + eps), where:
         - d_c is the mean L1 distance from j to cumulative means of the ordered neighbors.
         - m_c is either 1 (membership_mode="none") or the mean neighbor membership, where each neighbor's membership
           is the proportion ("soft") or hard majority ("hard") of same-label points in its own KNN over SOURCE.
      3) Split class mass across selected neighbors with a temperatured softmax over their raw L1 distances.
      4) Mix classes by pi_c = s_c / sum_c s_c and assign final edge weights gamma_{c,r} = pi_c * softmax_in_class[r].
    Returns:
        edge_index: int64 array of shape (2, E)
        edge_attr:  float32 array of shape (E,)
    """
    src_ids = np.asarray(source_indices, dtype=np.int64)
    dest_ids   = np.asarray(destination_indices, dtype=np.int64)
    label_vector = np.asarray(label_vector)

    # Map each class to its source ids for quick lookups
    src_labels = label_vector[src_ids]
    classes = np.unique(src_labels)
    sources_by_class: Dict[int, np.ndarray] = {c: src_ids[src_labels == c] for c in classes}

    # --- Helpers ----------------------------------------------------------------
    mem_cache: Dict[int, float] = {}

    def neighbor_membership(i: int) -> float:
        """Membership of source i w.r.t. its own KNN over SOURCE ids."""
        if membership_mode == "none":
            return 1.0
        if i in mem_cache:
            return mem_cache[i]

        xi = decision_matrix[i]
        pool = src_ids[src_ids != i]
        if pool.size == 0:
            mem_cache[i] = 1.0
            return 1.0

        d = np.sum(np.abs(decision_matrix[pool] - xi), axis=1)
        k = min(int(membership_k), pool.size)
        nn = pool[np.argpartition(d, k - 1)[:k]] if k > 0 else np.empty(0, dtype=np.int64)
        prop_same = float(np.mean(label_vector[nn] == label_vector[i])) if nn.size else 1.0

        val = 1.0 if (membership_mode == "hard" and prop_same > 0.5) else (prop_same if membership_mode == "soft" else 1.0)
        mem_cache[i] = val
        return val

    def softmax_over_neg(dist: np.ndarray) -> np.ndarray:
        """Compute softmax over -dist with temperature = median(dist)."""
        tau = max(float(np.median(dist)), eps)
        z = (-dist / tau)
        z -= z.max()  # numerical stability
        w = np.exp(z)
        return w / w.sum()

    # --- Build edges ------------------------------------------------------------
    src_list, dst_list, w_list = [], [], []

    for j in dest_ids.tolist():
        q = decision_matrix[j]

        # First pass: collect per-class neighbors and CMDW scores s_c
        per_class_neighbors: Dict[int, np.ndarray] = {}
        per_class_raw_dists: Dict[int, np.ndarray] = {}
        class_scores: Dict[int, float] = {}

        for c in classes.tolist():
            S_c = sources_by_class[c]
            if label_vector[j] == c:
                S_c = S_c[S_c != j]  # avoid self if same class
            if S_c.size == 0:
                continue

            # k nearest within class
            d_all = np.sum(np.abs(decision_matrix[S_c] - q), axis=1)
            k_c = min(int(k_per_class), S_c.size)
            idx = np.argpartition(d_all, k_c - 1)[:k_c]
            neigh_ids = S_c[idx]           # (k_c,)
            neigh_d   = d_all[idx]         # (k_c,)
            per_class_neighbors[c] = neigh_ids
            per_class_raw_dists[c] = neigh_d

            # CMDW: mean distance to cumulative means
            cum_means = np.cumsum(decision_matrix[neigh_ids], axis=0) / np.arange(1, k_c + 1)[:, None]
            dbar_c = float(np.mean(np.sum(np.abs(cum_means - q), axis=1)))

            # Membership average over neighbors
            mbar_c = 1.0 if membership_mode == "none" else float(np.mean([neighbor_membership(int(i)) for i in neigh_ids]))
            class_scores[c] = mbar_c / (dbar_c + eps)

        if not class_scores:
            continue

        # Normalize class masses pi_c
        total_score = float(sum(class_scores.values()))
        class_mass = {c: class_scores[c] / total_score for c in class_scores}

        # Second pass: within-class softmax + mix with pi_c
        for c, neigh_ids in per_class_neighbors.items():
            w_in_class = softmax_over_neg(per_class_raw_dists[c]).astype(np.float32)
            gamma = (class_mass[c] * w_in_class).astype(np.float32)

            src_list.extend(neigh_ids.tolist())
            dst_list.extend([j] * neigh_ids.size)
            w_list.extend(gamma.tolist())

    if not src_list:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    if log_class_edge_stats:
        pair_weights: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for src_id, dst_id, weight in zip(src_list, dst_list, w_list):
            pair_weights[(int(label_vector[dst_id]), int(label_vector[src_id]))].append(weight)

        classes = np.unique(label_vector)
        for dst_cls in classes.tolist():
            for src_cls in classes.tolist():
                weights = pair_weights.get((int(dst_cls), int(src_cls)), [])
                avg_weight = float(np.mean(weights)) if weights else 0.0
                print(f"[SS edges] class {int(dst_cls)} incoming from {int(src_cls)} average weight: {avg_weight:.6f}")

    edge_index = np.vstack([np.asarray(src_list, dtype=np.int64),
                            np.asarray(dst_list, dtype=np.int64)])
    edge_attr = np.asarray(w_list, dtype=np.float32)
    return edge_index, edge_attr


def _to_numpy(x, dtype=None):
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            import torch
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def build_cs_edges(
    *,
    tr_meta_labels: np.ndarray,      # [n_train, M] correctness (0/1)
    decision_all: np.ndarray,        # [n_total, M*C] flattened probs
    y_train: np.ndarray,             # [n_train]
    ss_edge_index: np.ndarray,       # [2, E] sample->sample
    ss_edge_attr: np.ndarray,        # [E] weights
    n_train: int,
    n_total: int,
    num_classes: int,
    top_k: int = 3,
    score_mode: str = "balanced_acc",
    tie_break_mode: Optional[str] = "true_prob",
    gain_baseline: str = "mean_pool",
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FedDES CS edge builder with configurable score/tie-break modes.

    Score/tie-break modes:
      - score_mode: {"logloss", "gain", "balanced_gain", "balanced_acc", "true_prob"}
      - tie_break_mode: same set or None
    """
    tr_meta_labels = _to_numpy(tr_meta_labels, np.float32)
    decision_all   = _to_numpy(decision_all,   np.float32)
    y_train        = _to_numpy(y_train,        np.int64)
    ss_edge_index  = _to_numpy(ss_edge_index,  np.int64)
    ss_edge_attr   = _to_numpy(ss_edge_attr,   np.float32)

    M = int(tr_meta_labels.shape[1])
    C = int(num_classes)

    # ---------- build neighbor bundles ----------
    src, dst = ss_edge_index
    w = ss_edge_attr.astype(np.float32)

    neighbors_by_dest: Dict[int, Dict[str, List]] = {}

    for s, d, weight in zip(src.tolist(), dst.tolist(), w.tolist()):
        if s >= n_train: continue
        bundle = neighbors_by_dest.setdefault(d, {"idx": [], "w": []})
        bundle["idx"].append(s)
        bundle["w"].append(weight)

    # Convert lists to arrays for vectorized scoring
    final_bundles = {}
    for d, bundle in neighbors_by_dest.items():
        final_bundles[d] = {
            "idx": np.asarray(bundle["idx"], dtype=np.int64),
            "w":   np.asarray(bundle["w"],   dtype=np.float32)
        }

    # ---------- Internal Scoring Helpers ----------
    def neigh_true_prob_matrix(neigh_idx: np.ndarray) -> np.ndarray:
        neigh_label = y_train[neigh_idx]
        base = (np.arange(M, dtype=np.int64) * C)[None, :]
        idx  = base + neigh_label[:, None]
        return decision_all[neigh_idx[:, None], idx]

    def score_logloss(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        P = neigh_true_prob_matrix(neigh_idx)
        P = np.clip(P, eps, 1.0)
        return -(neigh_w[:, None] * np.log(P)).sum(axis=0)

    def score_gain(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        corr = tr_meta_labels[neigh_idx, :]
        if gain_baseline == "mean_pool":
            base = corr.mean(axis=1, keepdims=True)
        else:
            base = 0.0
        return (neigh_w[:, None] * (corr - base)).sum(axis=0)

    def score_balanced_gain(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        neigh_label = y_train[neigh_idx]
        classes = np.unique(neigh_label)
        corr = tr_meta_labels[neigh_idx, :].astype(np.float32)

        if gain_baseline == "mean_pool":
            base = corr.mean(axis=1, keepdims=True)
        else:
            base = 0.0
        lift = corr - base

        out = np.zeros(M, dtype=np.float32)
        denom_classes = 0

        for c in classes.tolist():
            mask = (neigh_label == c)
            if not np.any(mask):
                continue
            w_c = neigh_w[mask]
            denom = float(w_c.sum()) + eps
            g_c = (w_c[:, None] * lift[mask, :]).sum(axis=0) / denom
            out += g_c
            denom_classes += 1

        if denom_classes > 0:
            out /= float(denom_classes)
        return out

    def score_balanced_acc(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        neigh_label = y_train[neigh_idx]
        classes = np.unique(neigh_label)
        corr = tr_meta_labels[neigh_idx, :]
        out = np.zeros(M, dtype=np.float32)
        denom_classes = 0
        for c in classes:
            mask = (neigh_label == c)
            w_c = neigh_w[mask]
            denom = float(w_c.sum()) + eps
            a_c = (w_c[:, None] * corr[mask, :]).sum(axis=0) / denom
            out += a_c
            denom_classes += 1
        if denom_classes > 0: out /= denom_classes
        return out

    def score_true_prob(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        P = neigh_true_prob_matrix(neigh_idx)
        return (neigh_w[:, None] * P).sum(axis=0)

    def compute_score(mode: str, idx: np.ndarray, w: np.ndarray):
        if mode == "logloss": return score_logloss(idx, w)
        if mode == "gain": return score_gain(idx, w)
        if mode == "balanced_gain": return score_balanced_gain(idx, w)
        if mode == "balanced_acc": return score_balanced_acc(idx, w)
        if mode == "true_prob": return score_true_prob(idx, w)
        raise ValueError(f"Unknown mode: {mode}")

    # ---------- Main Scoring Loop ----------
    edge_src, edge_dst, edge_attr = [], [], []

    for dest_id in range(n_total):
        bundle = final_bundles.get(dest_id)
        if bundle is None or bundle["idx"].size == 0:
            continue

        idx, w_arr = bundle["idx"], bundle["w"]

        # Primary Score
        primary = compute_score(score_mode, idx, w_arr)

        # Secondary Score
        secondary = None
        if tie_break_mode:
            secondary = compute_score(tie_break_mode, idx, w_arr)

        # Sorting
        if score_mode == "logloss":
            pkey = primary
            if secondary is not None:
                skey = secondary if tie_break_mode == "logloss" else -secondary
                order = np.lexsort((skey, pkey))
            else:
                order = np.argsort(pkey)
        else:
            pkey = -primary
            if secondary is not None:
                skey = secondary if tie_break_mode == "logloss" else -secondary
                order = np.lexsort((skey, pkey))
            else:
                order = np.argsort(pkey)

        keep = order[:min(top_k, M)]

        # Edge Weight Calc (Monotone transformation)
        s = primary.astype(np.float32)
        if score_mode == "logloss":
            s0 = s - s.min()
            w_primary = np.exp(-s0)
        else:
            s0 = s - s.min()
            w_primary = s0 / (s0.max() + eps) if s0.max() > 0 else np.ones_like(s0)

        for clf_id in keep:
            edge_src.append(int(clf_id))
            edge_dst.append(int(dest_id))
            edge_attr.append(float(w_primary[clf_id]))

    if not edge_src:
        return np.zeros((2,0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    return np.vstack([edge_src, edge_dst]), np.array(edge_attr, dtype=np.float32)

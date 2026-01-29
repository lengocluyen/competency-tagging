"""TF-IDF + One-vs-Rest Logistic Regression multi-label baseline.

Implements user-requested options:
- word ngrams (1-2 or 1-3), min_df, max_features
- char ngrams (3-5) via analyzer=char_wb

Outputs `logreg_results.csv` with fold/UV metrics (same schema style as other baselines).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import data_io
import metrics as metrics_module
from aggregate import ResourceAggregator


def compute_mrr_from_scores(
    scores: np.ndarray,
    frag_ids: List[str],
    gold: Dict[str, Dict[str, Any]],
    comp_ids: List[str],
    *,
    at_k: Optional[int] = None,
) -> float:
    """Compute mean reciprocal rank (MRR) for multi-label ranking.

    For each fragment with at least one gold label, we rank all labels by
    descending score, then compute reciprocal rank of the first gold label.
    If at_k is set, we truncate the ranking to top-k (MRR@k).
    """
    rrs: List[float] = []

    for i, fid in enumerate(frag_ids):
        g = gold.get(fid) or {}
        gold_ids = {str(x.get("competency_id")) for x in (g.get("gold") or []) if x.get("competency_id")}
        if not gold_ids:
            continue

        row = scores[i]
        idx = np.argsort(-row)
        if at_k is not None and at_k > 0:
            idx = idx[:at_k]

        rr = 0.0
        for rank0, j in enumerate(idx):
            if comp_ids[int(j)] in gold_ids:
                rr = 1.0 / float(rank0 + 1)
                break
        rrs.append(rr)

    return float(np.mean(rrs) if rrs else 0.0)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_cv_splits(uv_list: List[str], n_folds: int = 5, seed: int = 7) -> List[Tuple[List[str], List[str]]]:
    rng = np.random.RandomState(seed)
    uv_array = np.array(uv_list)
    rng.shuffle(uv_array)

    splits: List[Tuple[List[str], List[str]]] = []
    fold_size = len(uv_array) // n_folds

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else len(uv_array)
        test_uvs = uv_array[test_start:test_end].tolist()
        train_uvs = [uv for uv in uv_list if uv not in test_uvs]
        splits.append((train_uvs, test_uvs))

    return splits


def build_label_mapping(competencies: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    comp_ids = sorted(competencies.keys())
    id2idx = {cid: i for i, cid in enumerate(comp_ids)}
    return id2idx, comp_ids


def load_fragments_and_gold_for_uvs(
    uvs: List[str],
    uv_to_frag: Dict[str, Path],
    uv_to_gold: Dict[str, Path],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    all_fragments: Dict[str, Dict[str, Any]] = {}
    all_gold: Dict[str, Dict[str, Any]] = {}

    for uv in uvs:
        all_fragments.update(data_io.load_fragments(uv_to_frag[uv]))
        all_gold.update(data_io.load_gold_fragments(uv_to_gold[uv]))

    return all_fragments, all_gold


def build_training_matrix(
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    id2idx: Dict[str, int],
) -> Tuple[List[str], np.ndarray, List[str]]:
    texts: List[str] = []
    labels: List[np.ndarray] = []
    frag_ids: List[str] = []

    num_labels = len(id2idx)

    for frag_id, frag in fragments.items():
        text = frag.get("text", "")
        y = np.zeros(num_labels, dtype=np.float32)
        if frag_id in gold:
            for g in gold[frag_id].get("gold", []):
                cid = g.get("competency_id")
                if cid in id2idx:
                    y[id2idx[cid]] = 1.0
        texts.append(text)
        labels.append(y)
        frag_ids.append(frag_id)

    Y = np.stack(labels, axis=0) if labels else np.zeros((0, num_labels), dtype=np.float32)
    return texts, Y, frag_ids


def train_logreg(
    train_texts: List[str],
    train_labels: np.ndarray,
    *,
    analyzer: str,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_features: int,
    C: float,
) -> Tuple[TfidfVectorizer, OneVsRestClassifier]:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(train_texts)

    base = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=C,
        max_iter=300,
        n_jobs=-1,
    )
    clf = OneVsRestClassifier(base)
    clf.fit(X, train_labels)

    return vectorizer, clf


def predict_logreg(
    vectorizer: TfidfVectorizer,
    clf: OneVsRestClassifier,
    fragments: Dict[str, Dict[str, Any]],
    comp_ids: List[str],
    *,
    threshold: float,
    top_k: Optional[int],
) -> List[Dict[str, Any]]:
    frag_ids: List[str] = []
    texts: List[str] = []

    for fid, frag in fragments.items():
        frag_ids.append(fid)
        texts.append(frag.get("text", ""))

    if not texts:
        return []

    X = vectorizer.transform(texts)

    # OneVsRest(LogReg) supports predict_proba
    probs = clf.predict_proba(X)  # shape [n, n_labels]

    out: List[Dict[str, Any]] = []
    for fid, row in zip(frag_ids, probs):
        if top_k is not None and top_k > 0:
            idx = np.argsort(-row)[:top_k]
            preds = [{"competency_id": comp_ids[i], "confidence": float(row[i])} for i in idx]
        else:
            idx = np.where(row >= threshold)[0]
            preds = [{"competency_id": comp_ids[i], "confidence": float(row[i])} for i in idx]
        out.append({"fragment_id": fid, "predictions": preds})

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-IDF + OVR Logistic Regression baseline")
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--gold_dir", type=Path, required=True)
    ap.add_argument("--competencies", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("output_logreg_alluvs"))
    ap.add_argument("--exclude_uv", action="append", default=[], help="UV id to exclude (repeatable)")

    ap.add_argument("--analyzer", choices=["word", "char_wb"], default="word")
    ap.add_argument("--word_ngrams", choices=["1-2", "1-3"], default="1-2")
    ap.add_argument("--char_ngrams", default="3-5", help="Used when analyzer=char_wb")
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--C", type=float, default=2.0)

    ap.add_argument("--pred_threshold", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    competencies = data_io.load_competencies(args.competencies)
    id2idx, comp_ids = build_label_mapping(competencies)

    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)
    uv_to_frag = {data_io.get_uv_from_path(p): p for p in fragment_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    exclude = {str(u).strip() for u in args.exclude_uv}
    uvs = sorted(set(uv_to_frag.keys()) & set(uv_to_gold.keys()))
    uvs = [u for u in uvs if u not in exclude]
    n_folds = int(config.get("n_folds", 5))

    cv_splits = create_cv_splits(uvs, n_folds=n_folds, seed=args.seed)

    if args.analyzer == "word":
        ngram_range = (1, 2) if args.word_ngrams == "1-2" else (1, 3)
    else:
        lo, hi = args.char_ngrams.split("-")
        ngram_range = (int(lo), int(hi))

    rows: List[Dict[str, Any]] = []

    for fold_idx, (train_uvs, test_uvs) in enumerate(cv_splits):
        train_frags, train_gold = load_fragments_and_gold_for_uvs(train_uvs, uv_to_frag, uv_to_gold)
        train_texts, train_Y, _ = build_training_matrix(train_frags, train_gold, id2idx)

        vectorizer, clf = train_logreg(
            train_texts,
            train_Y,
            analyzer=args.analyzer,
            ngram_range=ngram_range,
            min_df=args.min_df,
            max_features=args.max_features,
            C=args.C,
        )

        for uv in test_uvs:
            fragments = data_io.load_fragments(uv_to_frag[uv])
            gold = data_io.load_gold_fragments(uv_to_gold[uv])

            # Build deterministic fragment order for both predictions and MRR.
            frag_ids = list(fragments.keys())
            texts = [fragments[fid].get("text", "") for fid in frag_ids]
            X_uv = vectorizer.transform(texts) if texts else None
            probs_uv = clf.predict_proba(X_uv) if X_uv is not None else np.zeros((0, len(comp_ids)), dtype=np.float32)

            mrr = compute_mrr_from_scores(probs_uv, frag_ids, gold, comp_ids)
            mrr_at_k = compute_mrr_from_scores(probs_uv, frag_ids, gold, comp_ids, at_k=args.top_k)

            frag_preds = predict_logreg(
                vectorizer,
                clf,
                fragments,
                comp_ids,
                threshold=args.pred_threshold,
                top_k=args.top_k,
            )

            reconcile_stats = [
                {"parent_child_redundancy": 0, "prereq_violations": 0, "capped_by_max_labels": 0}
                for _ in frag_preds
            ]

            aggregator = ResourceAggregator(
                aggregation_method=config.get("aggregation_method", "max"),
                fragment_type_weights=config.get("fragment_type_weights", {}),
                top_k_per_resource=config.get("top_k_per_resource", 10),
                threshold=float(config.get("supervised_aggregation_threshold", 0.0)),
            )

            res_preds = aggregator.aggregate(frag_preds, fragments)

            m = metrics_module.evaluate_pipeline(
                fragment_predictions=frag_preds,
                resource_predictions=res_preds,
                gold=gold,
                fragments=fragments,
                reconcile_stats=reconcile_stats,
            )

            rows.append({"fold": fold_idx, "uv": uv, "mrr": mrr, "mrr_at_k": mrr_at_k, **m})

    out_csv = args.output_dir / "logreg_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

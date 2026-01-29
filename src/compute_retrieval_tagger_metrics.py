"""Compute end-to-end tagging metrics for retrieval-only methods.

Why this exists:
- retrieval_eval.py computes retrieval metrics (Hit@K/Recall@K/MRR) only.
- The consolidated table expects tagging metrics (micro/macro F1, resource macro-F1, evidence valid).

This script turns a retrieval ranking into a multi-label tagger by predicting the
Top-K retrieved competency IDs per fragment (optionally filtering non-positive
scores), then evaluates using metrics.py.

Data note:
- We use gold fragment files as the source of fragment text + resource_id so this
  works even when the raw fragment collection directory is not available.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

import data_io
import metrics
from retrieval_eval import (
    Ranking,
    build_comp_text,
    rank_bm25,
    rank_dense,
    rrf_hybrid,
    weighted_sum_hybrid,
)


def _build_fragment_predictions(
    frag_ids: List[str],
    rankings: List[Ranking],
    *,
    pred_k: int,
    filter_nonpositive: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fid, r in zip(frag_ids, rankings):
        comps: List[str] = []
        if pred_k > 0:
            if filter_nonpositive:
                for cid, s in zip(r.comp_ids[:pred_k], r.scores[:pred_k]):
                    if float(s) > 0.0:
                        comps.append(str(cid))
            else:
                comps = [str(cid) for cid in r.comp_ids[:pred_k]]

        out.append({"fragment_id": fid, "predictions": [{"competency_id": c} for c in comps]})
    return out


def _build_resource_predictions(
    fragments: Dict[str, Dict[str, Any]],
    fragment_predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    res_to_labels: Dict[str, set[str]] = {}

    for fp in fragment_predictions:
        fid = fp["fragment_id"]
        res_id = (fragments.get(fid) or {}).get("resource_id")
        if not res_id:
            continue
        s = res_to_labels.setdefault(str(res_id), set())
        for p in fp.get("predictions", []):
            cid = p.get("competency_id")
            if cid:
                s.add(str(cid))

    return [
        {"resource_id": rid, "predictions": [{"competency_id": cid} for cid in sorted(cids)]}
        for rid, cids in res_to_labels.items()
    ]


def _evaluate_method_for_uv(
    *,
    method: str,
    frag_ids: List[str],
    frag_texts: List[str],
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    comp_ids: List[str],
    comp_texts: List[str],
    dense_model: str,
    pred_k: int,
    retrieval_k: int,
    filter_nonpositive: bool,
) -> Dict[str, Any]:
    if method == "bm25":
        rankings = rank_bm25(frag_texts, comp_ids, comp_texts, top_k=retrieval_k)
    elif method == "dense":
        rankings = rank_dense(
            frag_texts,
            comp_ids,
            comp_texts,
            model_name=dense_model,
            top_k=retrieval_k,
        )
    elif method in {"hybrid_rrf", "hybrid_weighted"}:
        bm25_rankings = rank_bm25(frag_texts, comp_ids, comp_texts, top_k=retrieval_k)
        dense_rankings = rank_dense(
            frag_texts,
            comp_ids,
            comp_texts,
            model_name=dense_model,
            top_k=retrieval_k,
        )
        if method == "hybrid_rrf":
            rankings = [rrf_hybrid(a, b, top_k=retrieval_k) for a, b in zip(bm25_rankings, dense_rankings)]
        else:
            rankings = [weighted_sum_hybrid(a, b, top_k=retrieval_k) for a, b in zip(bm25_rankings, dense_rankings)]
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

    fragment_predictions = _build_fragment_predictions(
        frag_ids,
        rankings,
        pred_k=pred_k,
        filter_nonpositive=filter_nonpositive,
    )
    resource_predictions = _build_resource_predictions(fragments, fragment_predictions)

    m = metrics.evaluate_pipeline(
        fragment_predictions=fragment_predictions,
        resource_predictions=resource_predictions,
        gold=gold,
        fragments=fragments,
        reconcile_stats=[],
    )

    return {
        "method": method,
        "pred_k": int(pred_k),
        "micro_f1": float(m.get("micro_f1", 0.0)),
        "macro_f1": float(m.get("macro_f1", 0.0)),
        "resource_macro_f1": float(m.get("resource_macro_f1", 0.0)),
        "evidence_valid_rate": float(m.get("evidence_valid_rate", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute tagging metrics for retrieval methods (bm25/dense/hybrid).")
    ap.add_argument("--gold_dir", type=Path, default=Path("golds_fragments"))
    ap.add_argument("--competencies", type=Path, default=Path("competencies_utc.jsonl"))
    ap.add_argument("--exclude_uv", action="append", default=[])

    ap.add_argument("--methods", default="bm25,dense,hybrid_rrf,hybrid_weighted")
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pred_k", type=int, default=20, help="How many top retrieved labels to predict per fragment")
    ap.add_argument(
        "--retrieval_k",
        type=int,
        default=50,
        help="How many labels to retrieve before taking top pred_k (must be >= pred_k)",
    )
    ap.add_argument(
        "--filter_nonpositive",
        action="store_true",
        help="Drop retrieved labels with non-positive scores (enables predicting NONE)",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=Path("output_retrieval_eval") / "retrieval_tagger_results.csv",
    )

    args = ap.parse_args()

    exclude = {str(u).strip() for u in args.exclude_uv}
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    if args.retrieval_k < args.pred_k:
        raise SystemExit("--retrieval_k must be >= --pred_k")

    competencies = data_io.load_competencies(args.competencies)
    comp_ids = sorted(competencies.keys())
    comp_texts = [build_comp_text(competencies[cid]) or cid for cid in comp_ids]

    gold_files = data_io.find_gold_files(args.gold_dir)
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}
    uvs = sorted(uv_to_gold.keys())

    rows_out: List[Dict[str, Any]] = []

    for uv in uvs:
        if uv in exclude:
            continue

        gold = data_io.load_gold_fragments(uv_to_gold[uv])
        fragments = gold  # gold records contain text/resource_id, sufficient for evaluation

        frag_ids = list(gold.keys())
        frag_texts = [gold[fid].get("text", "") for fid in frag_ids]

        if not frag_ids:
            continue

        for method in methods:
            row = _evaluate_method_for_uv(
                method=method,
                frag_ids=frag_ids,
                frag_texts=frag_texts,
                fragments=fragments,
                gold=gold,
                comp_ids=comp_ids,
                comp_texts=comp_texts,
                dense_model=args.dense_model,
                pred_k=args.pred_k,
                retrieval_k=args.retrieval_k,
                filter_nonpositive=bool(args.filter_nonpositive),
            )
            rows_out.append({"uv": uv, **row})

    out_path = args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "uv",
        "method",
        "pred_k",
        "micro_f1",
        "macro_f1",
        "resource_macro_f1",
        "evidence_valid_rate",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print("Wrote:", out_path)
    if rows_out:
        # Quick sanity summary
        by_method: Dict[str, List[float]] = {}
        for r in rows_out:
            by_method.setdefault(str(r["method"]), []).append(float(r["micro_f1"]))
        for m in sorted(by_method.keys()):
            vals = np.asarray(by_method[m], dtype=np.float32)
            print(f"{m}: mean micro_f1={float(vals.mean()):.4f} (n_uv={len(vals)})")


if __name__ == "__main__":
    main()

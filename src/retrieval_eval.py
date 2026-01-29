"""Candidate generator evaluation: Recall@K / Hit@K / MRR.

Implements baselines:
- Dense retrieval bi-encoder (sentence-transformers)
- BM25 retrieval over competency profile texts
- Hybrid BM25 + Dense via RRF or weighted sum

This is retrieval-only evaluation (no tagging thresholding). It evaluates how
well the candidate generator retrieves gold competencies.

Outputs: output_retrieval_eval/retrieval_eval_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import data_io


def build_comp_text(comp: Dict[str, Any]) -> str:
    parts: List[str] = []
    label = comp.get("label") or comp.get("label_en") or comp.get("label_fr")
    if label:
        parts.append(str(label))
    desc = comp.get("description") or comp.get("description_en") or comp.get("description_fr")
    if desc:
        parts.append(str(desc))
    aliases = comp.get("aliases") or comp.get("aliases_en") or comp.get("aliases_fr") or []
    if isinstance(aliases, list) and aliases:
        parts.append(", ".join(map(str, aliases)))
    keywords = comp.get("keywords") or comp.get("keywords_en") or comp.get("keywords_fr") or []
    if isinstance(keywords, list) and keywords:
        parts.append(", ".join(map(str, keywords)))
    return ". ".join([p for p in parts if p]).strip()


def tokenize(text: str) -> List[str]:
    return [t for t in (text or "").lower().split() if t]


@dataclass
class Ranking:
    comp_ids: List[str]
    scores: np.ndarray


def rank_bm25(
    fragments: List[str],
    comp_ids: List[str],
    comp_texts: List[str],
    *,
    top_k: int,
) -> List[Ranking]:
    from rank_bm25 import BM25Okapi

    corpus_tokens = [tokenize(t) for t in comp_texts]
    bm25 = BM25Okapi(corpus_tokens)

    out: List[Ranking] = []
    for f in fragments:
        q = tokenize(f)
        scores = np.asarray(bm25.get_scores(q), dtype=np.float32)
        idx = np.argsort(-scores)[:top_k]
        out.append(Ranking(comp_ids=[comp_ids[i] for i in idx], scores=scores[idx]))
    return out


def rank_dense(
    fragments: List[str],
    comp_ids: List[str],
    comp_texts: List[str],
    *,
    model_name: str,
    top_k: int,
) -> List[Ranking]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    comp_emb = model.encode(comp_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    frag_emb = model.encode(fragments, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

    out: List[Ranking] = []
    for v in frag_emb:
        scores = np.dot(comp_emb, v)
        idx = np.argsort(-scores)[:top_k]
        out.append(Ranking(comp_ids=[comp_ids[i] for i in idx], scores=scores[idx].astype(np.float32)))
    return out


def rrf_hybrid(a: Ranking, b: Ranking, *, k0: int = 60, top_k: int = 50) -> Ranking:
    # Rank-based fusion; ignore raw scores.
    scores: Dict[str, float] = {}
    for r, cid in enumerate(a.comp_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k0 + r)
    for r, cid in enumerate(b.comp_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k0 + r)

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return Ranking(comp_ids=[cid for cid, _ in items], scores=np.asarray([s for _, s in items], dtype=np.float32))


def weighted_sum_hybrid(a: Ranking, b: Ranking, *, wa: float = 0.5, wb: float = 0.5, top_k: int = 50) -> Ranking:
    # Score-based fusion; normalize to [0,1] within each list.
    scores: Dict[str, float] = {}

    def norm(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo, hi = float(x.min()), float(x.max())
        if abs(hi - lo) < 1e-12:
            return np.ones_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    na = norm(a.scores)
    nb = norm(b.scores)

    for cid, s in zip(a.comp_ids, na):
        scores[cid] = scores.get(cid, 0.0) + wa * float(s)
    for cid, s in zip(b.comp_ids, nb):
        scores[cid] = scores.get(cid, 0.0) + wb * float(s)

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return Ranking(comp_ids=[cid for cid, _ in items], scores=np.asarray([s for _, s in items], dtype=np.float32))


def gold_labels_for_fragment(gold: Dict[str, Dict[str, Any]], frag_id: str) -> List[str]:
    g = gold.get(frag_id) or {}
    return [str(x.get("competency_id")) for x in (g.get("gold") or []) if x.get("competency_id")]


def eval_rankings(
    frag_ids: List[str],
    gold: Dict[str, Dict[str, Any]],
    rankings: List[Ranking],
    *,
    ks: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k in ks:
        hit = []
        recall = []
        rr = []

        for fid, r in zip(frag_ids, rankings):
            g = gold_labels_for_fragment(gold, fid)
            gset = set(g)
            if not gset:
                continue

            top = r.comp_ids[:k]
            top_set = set(top)

            hit.append(1.0 if (top_set & gset) else 0.0)
            recall.append(len(top_set & gset) / max(1, len(gset)))

            best_rank = None
            for i, cid in enumerate(r.comp_ids, start=1):
                if cid in gset:
                    best_rank = i
                    break
            rr.append(1.0 / best_rank if best_rank else 0.0)

        rows.append(
            {
                "k": k,
                "hit_at_k": float(np.mean(hit) if hit else 0.0),
                "recall_at_k": float(np.mean(recall) if recall else 0.0),
                "mrr": float(np.mean(rr) if rr else 0.0),
                "n_fragments": int(len(hit)),
            }
        )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval candidate generators")
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--gold_dir", type=Path, required=True)
    ap.add_argument("--competencies", type=Path, required=True)
    ap.add_argument("--exclude_uv", action="append", default=[])

    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--bm25_k", type=int, default=50)
    ap.add_argument("--dense_k", type=int, default=50)
    ap.add_argument("--out_csv", type=Path, default=Path("output_retrieval_eval") / "retrieval_eval_results.csv")
    ap.add_argument("--ks", default="1,5,10,20,50")

    args = ap.parse_args()
    exclude = {str(u).strip() for u in args.exclude_uv}
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    competencies = data_io.load_competencies(args.competencies)
    comp_ids = sorted(competencies.keys())
    comp_texts = [build_comp_text(competencies[cid]) or cid for cid in comp_ids]

    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)
    uv_to_frag = {data_io.get_uv_from_path(p): p for p in fragment_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    uvs = sorted(set(uv_to_frag.keys()) & set(uv_to_gold.keys()))

    rows_out: List[Dict[str, Any]] = []

    for uv in uvs:
        if uv in exclude:
            continue

        fragments = data_io.load_fragments(uv_to_frag[uv])
        gold = data_io.load_gold_fragments(uv_to_gold[uv])

        frag_ids = list(fragments.keys())
        frag_texts = [fragments[fid].get("text", "") for fid in frag_ids]

        if not frag_ids:
            continue

        bm25_rankings = rank_bm25(frag_texts, comp_ids, comp_texts, top_k=max(ks + [args.bm25_k]))
        dense_rankings = rank_dense(
            frag_texts,
            comp_ids,
            comp_texts,
            model_name=args.dense_model,
            top_k=max(ks + [args.dense_k]),
        )

        hybrid_rrf = [rrf_hybrid(a, b, top_k=max(ks)) for a, b in zip(bm25_rankings, dense_rankings)]
        hybrid_ws = [weighted_sum_hybrid(a, b, top_k=max(ks)) for a, b in zip(bm25_rankings, dense_rankings)]

        for method, rankings in [
            ("bm25", bm25_rankings),
            ("dense", dense_rankings),
            ("hybrid_rrf", hybrid_rrf),
            ("hybrid_weighted", hybrid_ws),
        ]:
            for r in eval_rankings(frag_ids, gold, rankings, ks=ks):
                rows_out.append({"uv": uv, "method": method, **r})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uv", "method", "k", "hit_at_k", "recall_at_k", "mrr", "n_fragments"])
        w.writeheader()
        w.writerows(rows_out)

    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()

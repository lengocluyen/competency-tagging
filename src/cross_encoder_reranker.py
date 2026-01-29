"""BM25 candidate generation + Cross-Encoder reranker baseline.

Goal: stronger IR baseline for candidate generation.
- Candidates: BM25 over competency profile texts (K candidates per fragment)
- Rerank: CrossEncoder scores for (fragment, competency_text)
- Train reranker per fold using train UVs:
  positives = gold labels
  negatives = BM25 candidates not in gold (optionally subsampled)

Outputs retrieval metrics (Hit@K, Recall@K, MRR) on test UVs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import data_io
import metrics


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
class CandidateList:
    frag_id: str
    candidates: List[str]


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


def bm25_candidates(
    fragment_texts: Dict[str, str],
    comp_ids: List[str],
    comp_texts: List[str],
    *,
    k: int,
) -> List[CandidateList]:
    from rank_bm25 import BM25Okapi

    corpus_tokens = [tokenize(t) for t in comp_texts]
    bm25 = BM25Okapi(corpus_tokens)

    out: List[CandidateList] = []
    for fid, text in fragment_texts.items():
        scores = np.asarray(bm25.get_scores(tokenize(text)), dtype=np.float32)
        idx = np.argsort(-scores)[:k]
        out.append(CandidateList(frag_id=fid, candidates=[comp_ids[i] for i in idx]))
    return out


def gold_map(gold: Dict[str, Dict[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for fid, g in gold.items():
        out[fid] = {str(x.get("competency_id")) for x in (g.get("gold") or []) if x.get("competency_id")}
    return out


def eval_ranked(
    ranked_ids: List[List[str]],
    frag_ids: List[str],
    gold_by_frag: Dict[str, set[str]],
    ks: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k in ks:
        hit = []
        recall = []
        rr = []

        for fid, ranking in zip(frag_ids, ranked_ids):
            g = gold_by_frag.get(fid) or set()
            if not g:
                continue

            top = ranking[:k]
            top_set = set(top)
            hit.append(1.0 if (top_set & g) else 0.0)
            recall.append(len(top_set & g) / max(1, len(g)))

            best_rank = None
            for i, cid in enumerate(ranking, start=1):
                if cid in g:
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
    ap = argparse.ArgumentParser(description="BM25 + Cross-Encoder reranker baseline")
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--gold_dir", type=Path, required=True)
    ap.add_argument("--competencies", type=Path, required=True)
    ap.add_argument("--exclude_uv", action="append", default=[])

    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--candidate_k", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--neg_per_pos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ks", default="1,5,10,20,50")
    ap.add_argument("--out_csv", type=Path, default=Path("output_cross_encoder_reranker_alluvs") / "reranker_retrieval_results.csv")

    ap.add_argument("--tagger_pred_k", type=int, default=20, help="Top-K reranked labels to predict per fragment")
    ap.add_argument(
        "--tagger_filter_nonpositive",
        action="store_true",
        help="Drop reranked labels with non-positive scores (enables predicting NONE)",
    )
    ap.add_argument(
        "--tagger_out_csv",
        type=Path,
        default=Path("output_cross_encoder_reranker_alluvs") / "reranker_tagger_results.csv",
    )

    args = ap.parse_args()
    exclude = {str(u).strip() for u in args.exclude_uv}
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    rng = np.random.RandomState(args.seed)

    competencies = data_io.load_competencies(args.competencies)
    comp_ids = sorted(competencies.keys())
    comp_texts = {cid: (build_comp_text(competencies[cid]) or cid) for cid in comp_ids}
    comp_text_list = [comp_texts[cid] for cid in comp_ids]

    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)
    uv_to_frag = {data_io.get_uv_from_path(p): p for p in fragment_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    uvs = sorted(set(uv_to_frag.keys()) & set(uv_to_gold.keys()))
    uvs = [u for u in uvs if u not in exclude]

    splits = create_cv_splits(uvs, n_folds=5, seed=args.seed)

    from sentence_transformers import CrossEncoder
    from sentence_transformers.readers import InputExample
    from torch.utils.data import DataLoader

    rows_out: List[Dict[str, Any]] = []
    tagger_rows_out: List[Dict[str, Any]] = []

    for fold_idx, (train_uvs, test_uvs) in enumerate(splits):
        print(f"\n=== Fold {fold_idx} ===")
        # Build training pairs
        train_pairs: List[InputExample] = []

        for uv in train_uvs:
            fragments = data_io.load_fragments(uv_to_frag[uv])
            gold = data_io.load_gold_fragments(uv_to_gold[uv])
            gmap = gold_map(gold)

            frag_texts = {fid: (fragments[fid].get("text", "")) for fid in fragments.keys()}
            cands = bm25_candidates(frag_texts, comp_ids, comp_text_list, k=args.candidate_k)

            for cl in cands:
                fid = cl.frag_id
                pos = list(gmap.get(fid) or set())
                if not pos:
                    continue

                neg_pool = [c for c in cl.candidates if c not in gmap.get(fid, set())]

                for p in pos:
                    train_pairs.append(InputExample(texts=[frag_texts[fid], comp_texts[p]], label=1.0))
                    if neg_pool:
                        neg = rng.choice(neg_pool, size=min(args.neg_per_pos, len(neg_pool)), replace=False)
                        for n in neg:
                            train_pairs.append(InputExample(texts=[frag_texts[fid], comp_texts[str(n)]], label=0.0))

        print(f"Training pairs: {len(train_pairs)}")
        model = CrossEncoder(args.model, num_labels=1)

        if train_pairs:
            train_loader = DataLoader(train_pairs, shuffle=True, batch_size=args.batch_size)
            warmup_steps = int(0.1 * len(train_loader) * max(1, args.epochs))
            model.fit(
                train_dataloader=train_loader,
                epochs=args.epochs,
                warmup_steps=warmup_steps,
                show_progress_bar=True,
            )

        # Evaluate on test UVs
        for uv in test_uvs:
            fragments = data_io.load_fragments(uv_to_frag[uv])
            gold = data_io.load_gold_fragments(uv_to_gold[uv])
            gmap = gold_map(gold)

            frag_ids = list(fragments.keys())
            frag_texts = {fid: (fragments[fid].get("text", "")) for fid in frag_ids}
            cands = bm25_candidates(frag_texts, comp_ids, comp_text_list, k=args.candidate_k)

            ranked: List[List[str]] = []
            ranked_scores: List[np.ndarray] = []
            for cl in cands:
                fid = cl.frag_id
                pairs = [(frag_texts[fid], comp_texts[cid]) for cid in cl.candidates]
                scores = model.predict(pairs)
                scores = np.asarray(scores).reshape(-1)
                idx = np.argsort(-scores)
                ranked.append([cl.candidates[i] for i in idx])
                ranked_scores.append(scores[idx].astype(np.float32))

            for r in eval_ranked(ranked, [c.frag_id for c in cands], gmap, ks=ks):
                rows_out.append({"fold": fold_idx, "uv": uv, "method": "bm25+cross_encoder", **r})

            # Tagger-style metrics: convert reranked list into Top-K predicted labels.
            frag_ids = [c.frag_id for c in cands]
            fragment_predictions: List[Dict[str, Any]] = []
            for fid, ids, sc in zip(frag_ids, ranked, ranked_scores):
                chosen: List[str] = []
                if args.tagger_pred_k > 0:
                    if args.tagger_filter_nonpositive:
                        for cid, s in zip(ids[: args.tagger_pred_k], sc[: args.tagger_pred_k]):
                            if float(s) > 0.0:
                                chosen.append(str(cid))
                    else:
                        chosen = [str(cid) for cid in ids[: args.tagger_pred_k]]
                fragment_predictions.append(
                    {"fragment_id": fid, "predictions": [{"competency_id": cid} for cid in chosen]}
                )

            # Aggregate to resource-level labels.
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

            resource_predictions = [
                {"resource_id": rid, "predictions": [{"competency_id": cid} for cid in sorted(cids)]}
                for rid, cids in res_to_labels.items()
            ]

            m = metrics.evaluate_pipeline(
                fragment_predictions=fragment_predictions,
                resource_predictions=resource_predictions,
                gold=gold,
                fragments=fragments,
                reconcile_stats=[],
            )
            tagger_rows_out.append(
                {
                    "fold": fold_idx,
                    "uv": uv,
                    "method": "bm25+cross_encoder",
                    "pred_k": int(args.tagger_pred_k),
                    "micro_f1": float(m.get("micro_f1", 0.0)),
                    "macro_f1": float(m.get("macro_f1", 0.0)),
                    "resource_macro_f1": float(m.get("resource_macro_f1", 0.0)),
                    "evidence_valid_rate": float(m.get("evidence_valid_rate", 0.0)),
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "uv", "method", "k", "hit_at_k", "recall_at_k", "mrr", "n_fragments"])
        w.writeheader()
        w.writerows(rows_out)

    print("Wrote:", args.out_csv)

    args.tagger_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.tagger_out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "uv",
                "method",
                "pred_k",
                "micro_f1",
                "macro_f1",
                "resource_macro_f1",
                "evidence_valid_rate",
            ],
        )
        w.writeheader()
        w.writerows(tagger_rows_out)

    print("Wrote:", args.tagger_out_csv)


if __name__ == "__main__":
    main()

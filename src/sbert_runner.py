"""Sentence-BERT semantic similarity baseline for competency tagging.

For each fragment, this baseline:
- Encodes the fragment text with a sentence-transformers model.
- Encodes each competency description once.
- Computes cosine similarity between fragment and all competencies.
- Selects the top-K most similar competencies as predictions.

No training, no graph reconciliation, no evidence spans.
Uses the same ResourceAggregator + metrics.evaluate_pipeline as other baselines.
"""
import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

import data_io
import metrics as metrics_module
from aggregate import ResourceAggregator


def compute_mrr_from_ranking(ranked_ids: List[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    for rank0, cid in enumerate(ranked_ids):
        if cid in gold_ids:
            return 1.0 / float(rank0 + 1)
    return 0.0


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_competency_texts(competencies: Dict[str, Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for cid, comp in competencies.items():
        parts = []
        label = comp.get("label") or comp.get("label_en") or comp.get("label_fr")
        if label:
            parts.append(str(label))
        desc = comp.get("description") or comp.get("description_en") or comp.get("description_fr")
        if desc:
            parts.append(str(desc))
        keywords = comp.get("keywords") or comp.get("keywords_en") or comp.get("keywords_fr") or []
        if isinstance(keywords, list):
            parts.append(", ".join(map(str, keywords)))
        aliases = comp.get("aliases") or comp.get("aliases_en") or comp.get("aliases_fr") or []
        if isinstance(aliases, list):
            parts.append(", ".join(map(str, aliases)))
        text = ". ".join(p for p in parts if p)
        texts.append(text if text else str(cid))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentence-BERT semantic similarity baseline")
    parser.add_argument("--data_dir", type=Path, required=True, help="Fragment dir")
    parser.add_argument("--gold_dir", type=Path, required=True, help="Gold dir")
    parser.add_argument("--competencies", type=Path, required=True, help="Competencies JSON/JSONL")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML")
    parser.add_argument("--output_dir", type=Path, default=Path("output_sbert"), help="Output dir")

    args = parser.parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading competencies...")
    competencies = data_io.load_competencies(args.competencies)
    print(f"Loaded {len(competencies)} competencies")

    comp_ids = sorted(competencies.keys())
    comp_texts = build_competency_texts(competencies)

    model_name = config.get("sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = config.get("sbert_top_k", 10)

    print(f"Loading Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding competency texts...")
    comp_embeddings = model.encode(comp_texts, batch_size=32, convert_to_tensor=True, show_progress_bar=True)

    # Discover UVs
    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)
    uv_to_frag = {data_io.get_uv_from_path(p): p for p in fragment_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    uvs = sorted(set(uv_to_frag.keys()) & set(uv_to_gold.keys()))
    print(f"Found {len(uvs)} UVs: {uvs}")

    all_rows: List[Dict[str, Any]] = []

    for uv in uvs:
        print(f"\nEvaluating UV: {uv}")
        fragments = data_io.load_fragments(uv_to_frag[uv])
        gold = data_io.load_gold_fragments(uv_to_gold[uv])
        print(f"Loaded {len(fragments)} fragments, {len(gold)} gold annotations")

        frag_ids: List[str] = []
        frag_texts: List[str] = []
        for fid, frag in fragments.items():
            frag_ids.append(fid)
            frag_texts.append(frag.get("text", ""))

        if not frag_texts:
            print("No fragments for this UV; skipping")
            continue

        print("Encoding fragment texts and computing similarities...")
        frag_embeddings = model.encode(frag_texts, batch_size=32, convert_to_tensor=True, show_progress_bar=True)

        fragment_predictions: List[Dict[str, Any]] = []
        rrs: List[float] = []

        for fid, f_emb in tqdm(list(zip(frag_ids, frag_embeddings)), desc=f"Scoring {uv}"):
            sims = util.cos_sim(f_emb, comp_embeddings)[0]  # shape: [num_comp]
            sims_np = sims.cpu().numpy()
            ranked_idx = np.argsort(-sims_np)
            ranked_ids = [comp_ids[i] for i in ranked_idx]

            g = gold.get(fid) or {}
            gold_ids = {str(x.get("competency_id")) for x in (g.get("gold") or []) if x.get("competency_id")}
            if gold_ids:
                rrs.append(compute_mrr_from_ranking(ranked_ids, gold_ids))

            top_idx = np.argsort(-sims_np)[:top_k]
            preds = [
                {
                    "competency_id": comp_ids[i],
                    "confidence": float(sims_np[i]),
                }
                for i in top_idx
            ]
            fragment_predictions.append({"fragment_id": fid, "predictions": preds})

        reconcile_stats = [
            {
                "parent_child_redundancy": 0,
                "prereq_violations": 0,
                "capped_by_max_labels": 0,
            }
            for _ in fragment_predictions
        ]

        aggregator = ResourceAggregator(
            aggregation_method=config.get("aggregation_method", "max"),
            fragment_type_weights=config.get("fragment_type_weights", {}),
            top_k_per_resource=config.get("top_k_per_resource", 10),
            threshold=config.get("threshold", 0.5),
        )

        resource_predictions = aggregator.aggregate(fragment_predictions, fragments)

        metrics = metrics_module.evaluate_pipeline(
            fragment_predictions=fragment_predictions,
            resource_predictions=resource_predictions,
            gold=gold,
            fragments=fragments,
            reconcile_stats=reconcile_stats,
        )

        row = {"uv": uv, **metrics}
        row["mrr"] = float(np.mean(rrs) if rrs else 0.0)
        all_rows.append(row)
        print(f"Metrics for UV {uv}: {metrics}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_dir / "sbert_results.csv", index=False)
    print("\nSaved SBERT baseline results to:", args.output_dir / "sbert_results.csv")


if __name__ == "__main__":
    main()

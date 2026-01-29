"""Zero-shot LLM classification baseline (no BM25, no graph).

This runner compares a pure instruction-tuned LLM classifier against
fragment-level and resource-level gold labels.

It reuses the existing LLMTagger but passes all competencies as
candidates (no retrieval pre-filter) and skips graph reconciliation.
"""
import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
from tqdm import tqdm
import pandas as pd

import data_io
from llm_tagger import LLMTagger
from aggregate import ResourceAggregator
import metrics as metrics_module


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_zero_shot_for_uv(
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    competencies: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, float]:
    """Run zero-shot LLM tagging for a single UV.

    Args:
        fragments: Fragment records
        gold: Gold annotations
        competencies: Competency records
        config: Configuration dict
        output_dir: Directory to write predictions

    Returns:
        Metrics dictionary for this UV
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM tagger (reuse same model/config as main pipeline)
    tagger = LLMTagger(
        competencies=competencies,
        model=config.get("llm_model", "gpt-4o-mini"),
        cache_dir=output_dir / "cache",
        max_retries=config.get("max_retries", 5),
        validate_evidence=config.get("validate_evidence", True),
        repair_invalid=config.get("repair_invalid", True),
        api_key=config.get("openai_api_key"),
    )

    aggregator = ResourceAggregator(
        aggregation_method=config.get("aggregation_method", "max"),
        fragment_type_weights=config.get("fragment_type_weights", {}),
        top_k_per_resource=config.get("top_k_per_resource", 10),
        threshold=config.get("threshold", 0.5),
    )

    # Build full candidate list (all competencies, no retrieval filter)
    all_candidates = [(cid, 1.0) for cid in competencies.keys()]

    fragment_predictions: List[Dict[str, Any]] = []

    for frag_id, frag in tqdm(fragments.items(), desc="Zero-shot tagging"):
        text = frag.get("text", "")
        tagging_result = tagger.tag_fragment(text, all_candidates)

        fragment_predictions.append(
            {
                "fragment_id": frag_id,
                "candidates": all_candidates,
                "tagging_result": tagging_result,
                "predictions": tagging_result.get("selected", []),
            }
        )

    # No graph reconciliation: create empty stats
    reconcile_stats = [
        {"parent_child_redundancy": 0, "prereq_violations": 0, "capped_by_max_labels": 0}
        for _ in fragment_predictions
    ]

    resource_predictions = aggregator.aggregate(fragment_predictions, fragments)

    metrics = metrics_module.evaluate_pipeline(
        fragment_predictions=fragment_predictions,
        resource_predictions=resource_predictions,
        gold=gold,
        fragments=fragments,
        reconcile_stats=reconcile_stats,
    )

    # Save predictions
    data_io.save_jsonl(
        fragment_predictions, output_dir / "fragment_predictions_zero_shot.jsonl"
    )
    data_io.save_jsonl(
        resource_predictions, output_dir / "resource_predictions_zero_shot.jsonl"
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM baseline (no BM25, no graph)"
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Fragment dir")
    parser.add_argument("--gold_dir", type=Path, required=True, help="Gold dir")
    parser.add_argument(
        "--competencies", type=Path, required=True, help="Competencies JSON/JSONL"
    )
    parser.add_argument("--config", type=Path, required=True, help="Config YAML")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("output_zero_shot"), help="Output dir"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load competencies
    print("Loading competencies...")
    competencies = data_io.load_competencies(args.competencies)
    print(f"Loaded {len(competencies)} competencies")

    # Discover UVs
    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)

    uv_to_fragment = {data_io.get_uv_from_path(p): p for p in fragment_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    uvs = sorted(set(uv_to_fragment.keys()) & set(uv_to_gold.keys()))
    print(f"Found {len(uvs)} UVs: {uvs}")

    all_rows: List[Dict[str, Any]] = []

    for uv in uvs:
        print(f"\n=== Zero-shot for UV: {uv} ===")
        fragments = data_io.load_fragments(uv_to_fragment[uv])
        gold = data_io.load_gold_fragments(uv_to_gold[uv])
        print(f"Loaded {len(fragments)} fragments, {len(gold)} gold annotations")

        uv_out_dir = args.output_dir / uv
        metrics = run_zero_shot_for_uv(
            fragments=fragments,
            gold=gold,
            competencies=competencies,
            config=config,
            output_dir=uv_out_dir,
        )

        row = {"uv": uv, **metrics}
        all_rows.append(row)
        print(f"Metrics for {uv}: {metrics}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_dir / "zero_shot_results.csv", index=False)
    print("\nSaved zero-shot results to:", args.output_dir / "zero_shot_results.csv")


if __name__ == "__main__":
    main()

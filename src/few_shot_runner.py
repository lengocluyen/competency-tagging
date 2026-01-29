"""Few-shot LLM classification baseline (in-context learning, no BM25, no graph).

This runner compares a few-shot instruction-tuned LLM classifier against
fragment-level and resource-level gold labels.

It reuses the existing LLMTagger for API access and evidence validation,
but builds a custom prompt that includes a small set of labeled training
examples (few-shot demonstrations) drawn from other UVs.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
from tqdm import tqdm
import pandas as pd

import data_io
from llm_tagger import LLMTagger
from aggregate import ResourceAggregator
from retrieval import CompetencyRetriever
import metrics as metrics_module


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_demos(
    uvs: List[str],
    fragments_by_uv: Dict[str, Dict[str, Dict[str, Any]]],
    gold_by_uv: Dict[str, Dict[str, Dict[str, Any]]],
    competencies: Dict[str, Dict[str, Any]],
    exclude_uv: str,
    n_demos: int,
) -> List[Tuple[str, List[str]]]:
    """Build a list of (fragment_text, [competency_ids]) demos from all UVs except exclude_uv."""
    examples: List[Tuple[str, List[str]]] = []

    for uv in uvs:
        if uv == exclude_uv:
            continue
        fragments = fragments_by_uv.get(uv, {})
        gold = gold_by_uv.get(uv, {})
        for frag_id, gold_rec in gold.items():
            gold_labels = gold_rec.get("gold", [])
            comp_ids = [g.get("competency_id") for g in gold_labels if g.get("competency_id") in competencies]
            if not comp_ids:
                continue
            frag = fragments.get(frag_id)
            if not frag:
                continue
            text = frag.get("text", "")
            if not text.strip():
                continue
            examples.append((text, comp_ids))

    # Simple truncation to first n_demos; could be randomized if desired
    return examples[:n_demos]


def build_few_shot_prompt(
    fragment_text: str,
    candidates: List[Tuple[str, float]],
    demos: List[Tuple[str, List[str]]],
    competencies: Dict[str, Dict[str, Any]],
) -> str:
    """Build a few-shot prompt using labeled demo fragments.

    Demos only show correct competency IDs (and optional labels),
    while the target fragment must be answered in strict JSON format
    with evidence spans, as in the main LLMTagger.
    """
    prompt_parts: List[str] = []

    # Detect language of the target fragment to pick competency labels/aliases
    lang = data_io.detect_language(fragment_text)

    prompt_parts.append(
        "You are a competency tagging expert.\n"
        "You will first see some EXAMPLES with their correct competencies,\n"
        "then a NEW fragment where you must output JSON with selected competencies and evidence.\n"
        "Most realistic fragments will match at least one competency; use 'none' only when the text is clearly unrelated to all candidates.\n"
    )

    if demos:
        for idx, (demo_text, demo_comp_ids) in enumerate(demos, start=1):
            prompt_parts.append(f"\nEXAMPLE {idx}:\n")
            prompt_parts.append("TEXT FRAGMENT (EXAMPLE):\n")
            prompt_parts.append(demo_text + "\n")
            prompt_parts.append("GOLD COMPETENCIES (IDs and labels):\n")
            for cid in demo_comp_ids:
                comp = competencies.get(cid, {})
                label = data_io.get_competency_label_for_language(comp, lang)
                prompt_parts.append(f"- {cid}: {label}\n")

    # Now the actual task with candidate competencies listed explicitly
    prompt_parts.append("\nNOW TAG THE FOLLOWING NEW FRAGMENT.\n")
    prompt_parts.append("TEXT FRAGMENT:\n")
    prompt_parts.append(fragment_text + "\n")

    prompt_parts.append("\nCANDIDATE COMPETENCIES:\n")
    for comp_id, _ in candidates:
        comp = competencies[comp_id]
        label = data_io.get_competency_label_for_language(comp, lang)
        desc = comp.get("description", "")[:200]
        prompt_parts.append(f"- {comp_id}: {label}\n  {desc}\n")

    prompt_parts.append(
        """\nINSTRUCTIONS:
1. Based on the EXAMPLES and the candidate list, identify which competencies are clearly demonstrated in the new fragment.
2. For each identified competency, extract the specific quote that provides evidence.
3. Provide the exact character positions (start, end) of the quote in the fragment.
4. Provide a confidence score (0.0-1.0) for each competency.
5. Only if the fragment is clearly unrelated to all candidate competencies, set "none": true; otherwise, select at least one competency that best matches the fragment.

OUTPUT FORMAT (strict JSON):
{
  "selected": [
    {
      "competency_id": "...",
      "confidence": 0.85,
      "evidence": {
        "quote": "exact text from fragment",
        "start_char": 0,
        "end_char": 100
      }
    }
  ],
  "none": false
}

Respond ONLY with valid JSON, no other text."""
    )

    return "".join(prompt_parts)


def few_shot_tag_fragment(
    tagger: LLMTagger,
    fragment_text: str,
    candidates: List[Tuple[str, float]],
    demos: List[Tuple[str, List[str]]],
) -> Dict[str, Any]:
    """Tag a fragment using a few-shot prompt and LLMTagger's API client.

    This mirrors LLMTagger.tag_fragment, but uses a custom prompt that
    includes in-context examples instead of the default single-fragment prompt.
    """
    prompt = build_few_shot_prompt(
        fragment_text=fragment_text,
        candidates=candidates,
        demos=demos,
        competencies=tagger.competencies,
    )

    response_text = tagger._call_api(prompt)  # type: ignore[attr-defined]

    # Parse response
    try:
        result: Dict[str, Any] = json.loads(response_text)  # type: ignore[name-defined]
    except Exception:
        result = {"selected": [], "none": True}

    # Validate and repair evidence using same logic as LLMTagger
    if getattr(tagger, "validate_evidence", False) and "selected" in result:
        validated_selected: List[Dict[str, Any]] = []
        candidate_ids = {c[0] for c in candidates}

        for item in result["selected"]:
            comp_id = item.get("competency_id")
            if comp_id not in candidate_ids:
                continue

            evidence = item.get("evidence")
            if evidence and not tagger._validate_evidence(fragment_text, evidence):  # type: ignore[attr-defined]
                if getattr(tagger, "repair_invalid", False):
                    repaired = tagger._repair_evidence(fragment_text, evidence)  # type: ignore[attr-defined]
                    if repaired:
                        item["evidence"] = repaired
                    else:
                        continue
                else:
                    continue

            validated_selected.append(item)

        result["selected"] = validated_selected

    return result


def run_few_shot_for_uv(
    uv: str,
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    uvs: List[str],
    fragments_by_uv: Dict[str, Dict[str, Dict[str, Any]]],
    gold_by_uv: Dict[str, Dict[str, Dict[str, Any]]],
    competencies: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, float]:
    """Run few-shot LLM tagging for a single UV."""
    from json import dumps  # local import to avoid polluting top-level

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM tagger (same config as main pipeline)
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

    # BM25 retriever to provide a focused candidate set per fragment
    retriever = CompetencyRetriever(competencies)

    # Few-shot demos drawn from other UVs
    n_demos = int(config.get("few_shot_n_examples", 5))
    demos = build_demos(
        uvs=uvs,
        fragments_by_uv=fragments_by_uv,
        gold_by_uv=gold_by_uv,
        competencies=competencies,
        exclude_uv=uv,
        n_demos=n_demos,
    )

    if not demos:
        print(f"[WARN] No demos available for UV {uv}; falling back to zero-shot behaviour.")

    fragment_predictions: List[Dict[str, Any]] = []

    for frag_id, frag in tqdm(fragments.items(), desc=f"Few-shot tagging ({uv})"):
        text = frag.get("text", "")
        # Use BM25 to retrieve a focused candidate set for this fragment
        top_k = int(config.get("few_shot_top_k", 10))
        candidates = retriever.retrieve(text, top_k=top_k)

        if demos:
            tagging_result = few_shot_tag_fragment(tagger, text, candidates, demos)
        else:
            # Degenerate case: use default tagger (equivalent to zero-shot with BM25 candidates)
            tagging_result = tagger.tag_fragment(text, candidates)

        fragment_predictions.append(
            {
                "fragment_id": frag_id,
                "candidates": candidates,
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
        fragment_predictions, output_dir / "fragment_predictions_few_shot.jsonl"
    )
    data_io.save_jsonl(
        resource_predictions, output_dir / "resource_predictions_few_shot.jsonl"
    )

    # Also store metrics JSON for this UV (optional, convenient for debugging)
    with open(output_dir / "metrics_few_shot.json", "w", encoding="utf-8") as f:
        f.write(dumps(metrics, ensure_ascii=False, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Few-shot LLM baseline (in-context learning, no BM25, no graph)"
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Fragment dir")
    parser.add_argument("--gold_dir", type=Path, required=True, help="Gold dir")
    parser.add_argument(
        "--competencies", type=Path, required=True, help="Competencies JSON/JSONL"
    )
    parser.add_argument("--config", type=Path, required=True, help="Config YAML")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("output_few_shot"), help="Output dir"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load competencies
    print("Loading competencies...")
    competencies = data_io.load_competencies(args.competencies)
    print(f"Loaded {len(competencies)} competencies")

    # Discover UVs and load all fragments/golds upfront
    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)

    uv_to_fragment = {data_io.get_uv_from_path(p): p for p in fragment_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    uvs = sorted(set(uv_to_fragment.keys()) & set(uv_to_gold.keys()))
    print(f"Found {len(uvs)} UVs: {uvs}")

    fragments_by_uv: Dict[str, Dict[str, Dict[str, Any]]] = {}
    gold_by_uv: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for uv in uvs:
        fragments_by_uv[uv] = data_io.load_fragments(uv_to_fragment[uv])
        gold_by_uv[uv] = data_io.load_gold_fragments(uv_to_gold[uv])

    all_rows: List[Dict[str, Any]] = []

    for uv in uvs:
        print(f"\n=== Few-shot for UV: {uv} ===")
        fragments = fragments_by_uv[uv]
        gold = gold_by_uv[uv]
        print(f"Loaded {len(fragments)} fragments, {len(gold)} gold annotations")

        uv_out_dir = args.output_dir / uv
        metrics = run_few_shot_for_uv(
            uv=uv,
            fragments=fragments,
            gold=gold,
            uvs=uvs,
            fragments_by_uv=fragments_by_uv,
            gold_by_uv=gold_by_uv,
            competencies=competencies,
            config=config,
            output_dir=uv_out_dir,
        )

        row = {"uv": uv, **metrics}
        all_rows.append(row)
        print(f"Metrics for {uv}: {metrics}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_dir / "few_shot_results.csv", index=False)
    print("\nSaved few-shot results to:", args.output_dir / "few_shot_results.csv")


if __name__ == "__main__":
    main()

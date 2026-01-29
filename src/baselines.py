"""Baseline methods for competency tagging.

Provides simple comparison systems:
- NONE baseline: predict no competencies
- BM25-only baseline: use retrieval only, no LLM or graph
- Keyword baseline: match fragment text against competency labels/aliases
"""
from pathlib import Path
from typing import Dict, List, Any

from aggregate import ResourceAggregator
import metrics as metrics_module
from retrieval import CompetencyRetriever


def _make_empty_reconcile_stats(num_fragments: int) -> List[Dict[str, Any]]:
    """Create dummy reconciliation stats (no graph used).

    Args:
        num_fragments: Number of fragments

    Returns:
        List of stats dicts with zeros
    """
    return [
        {
            "parent_child_redundancy": 0,
            "prereq_violations": 0,
            "capped_by_max_labels": 0,
        }
        for _ in range(num_fragments)
    ]


def _init_aggregator(config: Dict[str, Any]) -> ResourceAggregator:
    """Initialize a ResourceAggregator from config."""
    return ResourceAggregator(
        aggregation_method=config.get("aggregation_method", "max"),
        fragment_type_weights=config.get("fragment_type_weights", {}),
        top_k_per_resource=config.get("top_k_per_resource", 10),
        threshold=config.get("threshold", 0.5),
    )


def none_baseline(
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """NONE baseline: predict no competencies for any fragment.

    Args:
        fragments: Fragment data
        gold: Gold annotations
        config: Configuration dictionary

    Returns:
        Metrics dictionary
    """
    fragment_predictions = []

    for frag_id in fragments.keys():
        fragment_predictions.append(
            {
                "fragment_id": frag_id,
                "predictions": [],
                "tagging_result": {"selected": [], "none": True},
            }
        )

    reconcile_stats = _make_empty_reconcile_stats(len(fragment_predictions))
    aggregator = _init_aggregator(config)
    resource_predictions = aggregator.aggregate(fragment_predictions, fragments)

    metrics = metrics_module.evaluate_pipeline(
        fragment_predictions=fragment_predictions,
        resource_predictions=resource_predictions,
        gold=gold,
        fragments=fragments,
        reconcile_stats=reconcile_stats,
    )

    return metrics


def bm25_baseline(
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    competencies: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    retrieval_k: int,
) -> Dict[str, float]:
    """BM25-only baseline: use retrieval scores only.

    For each fragment, we take the top-M retrieved competencies and
    treat them as predictions with confidence 1.0 (no evidence).

    Args:
        fragments: Fragment data
        gold: Gold annotations
        competencies: Competency records
        config: Configuration dictionary
        retrieval_k: Number of candidates to retrieve per fragment

    Returns:
        Metrics dictionary
    """
    retriever = CompetencyRetriever(competencies)
    top_m = config.get("bm25_baseline_top_m", 3)

    fragment_predictions: List[Dict[str, Any]] = []

    for frag_id, frag in fragments.items():
        text = frag.get("text", "")
        candidates = retriever.retrieve(text, top_k=retrieval_k)
        top_candidates = candidates[:top_m]

        preds = [
            {"competency_id": comp_id, "confidence": 1.0}
            for comp_id, _ in top_candidates
        ]

        fragment_predictions.append(
            {
                "fragment_id": frag_id,
                "candidates": candidates,
                "predictions": preds,
            }
        )

    reconcile_stats = _make_empty_reconcile_stats(len(fragment_predictions))
    aggregator = _init_aggregator(config)
    resource_predictions = aggregator.aggregate(fragment_predictions, fragments)

    metrics = metrics_module.evaluate_pipeline(
        fragment_predictions=fragment_predictions,
        resource_predictions=resource_predictions,
        gold=gold,
        fragments=fragments,
        reconcile_stats=reconcile_stats,
    )

    return metrics


def _build_competency_terms(competencies: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build lowercased term lists (labels/aliases) per competency.

    Args:
        competencies: Competency records

    Returns:
        Mapping competency_id -> list of terms
    """
    comp_terms: Dict[str, List[str]] = {}

    for comp_id, comp in competencies.items():
        terms: List[str] = []

        for field in ["label", "label_fr", "label_en"]:
            if field in comp and comp[field]:
                terms.append(str(comp[field]))

        for field in ["aliases", "aliases_fr", "aliases_en"]:
            if field in comp and comp[field]:
                # aliases fields are usually lists
                if isinstance(comp[field], list):
                    terms.extend([str(t) for t in comp[field]])
                else:
                    terms.append(str(comp[field]))

        # Normalize and deduplicate
        norm_terms = []
        seen = set()
        for t in terms:
            lt = t.strip().lower()
            if lt and lt not in seen:
                seen.add(lt)
                norm_terms.append(lt)

        comp_terms[comp_id] = norm_terms

    return comp_terms


def keyword_baseline(
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    competencies: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Keyword baseline: match fragment text against labels/aliases.

    A competency is predicted if any of its label/alias strings occurs
    (case-insensitive substring) in the fragment text.

    Args:
        fragments: Fragment data
        gold: Gold annotations
        competencies: Competency records
        config: Configuration dictionary (unused, for API symmetry)

    Returns:
        Metrics dictionary
    """
    comp_terms = _build_competency_terms(competencies)
    fragment_predictions: List[Dict[str, Any]] = []

    for frag_id, frag in fragments.items():
        text = frag.get("text", "")
        text_l = text.lower()

        preds = []
        for comp_id, terms in comp_terms.items():
            for term in terms:
                # Avoid extremely short terms which match too often
                if len(term) < 3:
                    continue
                if term in text_l:
                    preds.append({"competency_id": comp_id, "confidence": 1.0})
                    break  # one match is enough per competency

        fragment_predictions.append(
            {
                "fragment_id": frag_id,
                "predictions": preds,
            }
        )

    reconcile_stats = _make_empty_reconcile_stats(len(fragment_predictions))
    aggregator = _init_aggregator(config)
    resource_predictions = aggregator.aggregate(fragment_predictions, fragments)

    metrics = metrics_module.evaluate_pipeline(
        fragment_predictions=fragment_predictions,
        resource_predictions=resource_predictions,
        gold=gold,
        fragments=fragments,
        reconcile_stats=reconcile_stats,
    )

    return metrics


def run_baselines_for_uv(
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    competencies: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    retrieval_k: int,
    output_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Run all baselines for a single UV and configuration.

    Args:
        fragments: Fragment data
        gold: Gold annotations
        competencies: Competency records
        config: Configuration
        retrieval_k: Retrieval K used in main experiment (for BM25 baseline)
        output_dir: Optional directory (reserved for future use to save preds)

    Returns:
        List of result dicts, each with keys:
            - method: str
            - metrics: Dict[str, float]
    """
    results: List[Dict[str, Any]] = []

    # NONE baseline
    none_metrics = none_baseline(fragments, gold, config)
    results.append({"method": "none", "metrics": none_metrics})

    # BM25-only baseline
    bm25_metrics = bm25_baseline(fragments, gold, competencies, config, retrieval_k)
    results.append({"method": "bm25_only", "metrics": bm25_metrics})

    # Keyword baseline
    kw_metrics = keyword_baseline(fragments, gold, competencies, config)
    results.append({"method": "keyword", "metrics": kw_metrics})

    return results

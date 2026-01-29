"""Derive BM25+LLM (no graph) results from an existing main run.

Key idea
- The main pipeline stores BOTH:
  - `tagging_result.selected` (raw LLM output before graph reconciliation)
  - `predictions` (post-reconciliation output)

So we can isolate reconciliation without any new LLM calls by evaluating:
- No-graph: use `tagging_result.selected` as `predictions`
- Graph: use stored `predictions`

This script computes fold/UV metrics for the no-graph variant and writes a CSV
compatible with `recompute_methods_comparison.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import data_io
import metrics as metrics_module
from aggregate import ResourceAggregator


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _normalize_uv(uv: str) -> str:
    return (uv or "").strip()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_no_graph_fragment_predictions(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return predictions in the shape expected by metrics: fragment_id + predictions."""
    out: list[dict[str, Any]] = []
    for row in main_rows:
        tagging = row.get("tagging_result") or {}
        selected = tagging.get("selected") or []
        out.append({"fragment_id": row.get("fragment_id"), "predictions": selected})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive BM25+LLM (no graph) results from main outputs")
    ap.add_argument("--main_output_dir", type=Path, default=Path("output_main_alluvs"))
    ap.add_argument("--data_dir", type=Path, default=Path("..") / "resources_fragments")
    ap.add_argument("--gold_dir", type=Path, default=Path("golds_fragments") / "gold_fragments_all_uvs")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("output_bm25_llm_nograph_alluvs"))
    ap.add_argument("--exclude_uv", action="append", default=[], help="UV id to exclude (repeatable)")
    ap.add_argument(
        "--write_predictions",
        action="store_true",
        help="Also write derived per-UV prediction JSONL files (no-graph) next to the CSV.",
    )

    args = ap.parse_args()

    exclude = {_normalize_uv(u) for u in args.exclude_uv}

    config = load_config(args.config)
    # Force the aggregation threshold to match the selected tau.
    config = dict(config)
    config["threshold"] = float(args.tau)

    base = args.main_output_dir / f"k{args.k}_tau{args.tau}"
    if not base.exists():
        raise SystemExit(f"Missing main run folder: {base}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    aggregator = ResourceAggregator(
        aggregation_method=config.get("aggregation_method", "max"),
        fragment_type_weights=config.get("fragment_type_weights", {}),
        top_k_per_resource=config.get("top_k_per_resource", 10),
        threshold=config.get("threshold", 0.5),
    )

    rows: list[dict[str, Any]] = []

    for fold_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold")):
        fold_idx = int(fold_dir.name.replace("fold", ""))

        for uv_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
            uv = _normalize_uv(uv_dir.name)
            if not uv or uv in exclude:
                continue

            pred_path = uv_dir / f"{uv}_fragment_predictions.jsonl"
            if not pred_path.exists():
                continue

            fragments_path = args.data_dir / f"{uv}_fragments.jsonl"
            gold_path = args.gold_dir / f"{uv}_gold_fragments.jsonl"
            if not fragments_path.exists() or not gold_path.exists():
                # Skip UVs not in evaluation set.
                continue

            fragments = data_io.load_fragments(fragments_path)
            gold = data_io.load_gold_fragments(gold_path)

            main_fragment_preds = _read_jsonl(pred_path)
            no_graph_fragment_preds = build_no_graph_fragment_predictions(main_fragment_preds)

            resource_predictions = aggregator.aggregate(no_graph_fragment_preds, fragments)

            reconcile_stats = [
                {"parent_child_redundancy": 0, "prereq_violations": 0, "capped_by_max_labels": 0}
                for _ in no_graph_fragment_preds
            ]

            metrics = metrics_module.evaluate_pipeline(
                fragment_predictions=no_graph_fragment_preds,
                resource_predictions=resource_predictions,
                gold=gold,
                fragments=fragments,
                reconcile_stats=reconcile_stats,
            )

            rows.append({"fold": fold_idx, "uv": uv, **metrics})

            if args.write_predictions:
                out_uv_dir = args.out_dir / f"k{args.k}_tau{args.tau}" / fold_dir.name / uv
                out_uv_dir.mkdir(parents=True, exist_ok=True)
                _write_jsonl(no_graph_fragment_preds, out_uv_dir / f"{uv}_fragment_predictions_no_graph.jsonl")
                _write_jsonl(resource_predictions, out_uv_dir / f"{uv}_resource_predictions_no_graph.jsonl")

    # Write CSV
    import pandas as pd

    out_csv = args.out_dir / "bm25_llm_no_graph_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()

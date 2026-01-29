from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import data_io


def _gold_set(gold: Dict[str, Dict[str, Any]], frag_id: str) -> set[str]:
    g = gold.get(frag_id) or {}
    return {str(x.get("competency_id")) for x in (g.get("gold") or []) if x.get("competency_id")}


def _rr_from_ranking(ranked_ids: List[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    for rank0, cid in enumerate(ranked_ids):
        if cid in gold_ids:
            return 1.0 / float(rank0 + 1)
    return 0.0


def _unique_confidence(items: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cid, conf in items:
        cid = str(cid)
        conf = float(conf)
        if (cid not in out) or (conf > out[cid]):
            out[cid] = conf
    return out


def _build_llm_ranking(line: Dict[str, Any], *, use_selected: bool) -> List[str]:
    # 1) predicted/selected items ranked by confidence desc
    pred_items: List[Tuple[str, float]] = []
    if use_selected:
        selected = ((line.get("tagging_result") or {}).get("selected") or [])
        pred_items = [(x.get("competency_id"), x.get("confidence", 0.0)) for x in selected if x.get("competency_id")]
    else:
        preds = line.get("predictions") or []
        pred_items = [(x.get("competency_id"), x.get("confidence", 0.0)) for x in preds if x.get("competency_id")]

    pred_map = _unique_confidence([(cid, conf) for cid, conf in pred_items if cid])
    pred_ranked = [cid for cid, _ in sorted(pred_map.items(), key=lambda x: (-x[1], x[0]))]

    # 2) remaining candidates ranked by score desc
    cand_items = line.get("candidates") or []
    cand_map = _unique_confidence([(cid, score) for cid, score in cand_items if cid])
    cand_ranked = [cid for cid, _ in sorted(cand_map.items(), key=lambda x: (-x[1], x[0]))]

    seen = set()
    ranking: List[str] = []
    for cid in pred_ranked + cand_ranked:
        if cid not in seen:
            ranking.append(cid)
            seen.add(cid)
    return ranking


def mrr_from_fragment_jsonl(
    jsonl_paths: List[Path],
    gold_by_uv: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    use_selected: bool,
) -> Tuple[float, int]:
    rrs: List[float] = []
    n = 0

    for p in jsonl_paths:
        uv = data_io.get_uv_from_path(p)
        gold = gold_by_uv[uv]
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                fid = str(obj.get("fragment_id"))
                gset = _gold_set(gold, fid)
                if not gset:
                    continue
                ranking = _build_llm_ranking(obj, use_selected=use_selected)
                rrs.append(_rr_from_ranking(ranking, gset))
                n += 1

    return float(np.mean(rrs) if rrs else 0.0), n


def _find_best_k_tau(repo_dir: Path, exclude_uv: set[str]) -> Tuple[int, float]:
    df = pd.read_csv(repo_dir / "output_main_alluvs" / "all_results.csv")
    df = df[~df["uv"].astype(str).isin(exclude_uv)].copy()
    grouped = df.groupby(["retrieval_k", "threshold"], dropna=False).mean(numeric_only=True).reset_index()
    best = grouped.iloc[grouped["micro_f1"].idxmax()]
    return int(best["retrieval_k"]), float(best["threshold"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute MRR for LLM-based methods from saved fragment_predictions JSONL outputs.")
    ap.add_argument("--repo_dir", type=Path, default=Path("."))
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--gold_dir", type=Path, required=True)
    ap.add_argument("--exclude_uv", action="append", default=[])
    ap.add_argument("--out_csv", type=Path, default=Path("output_summary_alluvs") / "mrr_llm_methods.csv")
    args = ap.parse_args()

    repo_dir = args.repo_dir
    exclude = {str(u).strip() for u in args.exclude_uv}

    # Gold by UV (only for UVs we will evaluate)
    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    # Best K,tau for main runner
    k, tau = _find_best_k_tau(repo_dir, exclude)
    run_dir = repo_dir / "output_main_alluvs" / f"k{k}_tau{tau:.1f}"

    # Collect main-run fragment prediction jsonls (only test UV folders exist per fold)
    main_jsonls: List[Path] = sorted(run_dir.glob("fold*/**/*_fragment_predictions.jsonl"))
    main_jsonls = [p for p in main_jsonls if data_io.get_uv_from_path(p) not in exclude]

    gold_by_uv_main: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in main_jsonls:
        uv = data_io.get_uv_from_path(p)
        if uv in exclude:
            continue
        gold_by_uv_main[uv] = data_io.load_gold_fragments(uv_to_gold[uv])

    rows: List[Dict[str, Any]] = []

    mrr_graph, n_graph = mrr_from_fragment_jsonl(main_jsonls, gold_by_uv_main, use_selected=False)
    rows.append({"method": "LLM+BM25+Graph (best K,τ)", "mrr": mrr_graph, "n_fragments": n_graph, "k": k, "tau": tau})

    mrr_nograph, n_nograph = mrr_from_fragment_jsonl(main_jsonls, gold_by_uv_main, use_selected=True)
    rows.append({"method": "LLM+BM25 (no graph)", "mrr": mrr_nograph, "n_fragments": n_nograph, "k": k, "tau": tau})

    # Zero-shot and few-shot outputs: per UV fragment_predictions_*.jsonl
    for method_name, out_dir_name, file_name in [
        ("Zero-shot LLM", "output_zero_shot_alluvs", "fragment_predictions_zero_shot.jsonl"),
        ("Few-shot LLM", "output_few_shot_alluvs", "fragment_predictions_few_shot.jsonl"),
    ]:
        base = repo_dir / out_dir_name
        jsonls = sorted(base.glob(f"*/{file_name}"))
        jsonls = [p for p in jsonls if data_io.get_uv_from_path(p) not in exclude]

        gold_by_uv: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for p in jsonls:
            uv = data_io.get_uv_from_path(p)
            if uv in exclude:
                continue
            gold_by_uv[uv] = data_io.load_gold_fragments(uv_to_gold[uv])

        mrr, n = mrr_from_fragment_jsonl(jsonls, gold_by_uv, use_selected=False)
        rows.append({"method": method_name, "mrr": mrr, "n_fragments": n, "k": "", "tau": ""})

    out_df = pd.DataFrame(rows)
    out_path = args.out_csv if args.out_csv.is_absolute() else (repo_dir / args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print(out_df[["method", "mrr", "n_fragments"]].sort_values("mrr", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

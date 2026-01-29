"""Run all competency-tagging approaches and summarize results.

This script orchestrates:
- Main BM25 + LLM + graph pipeline (runner.py) + its baselines (BM25-only, keyword, NONE).
- Zero-shot LLM baseline (zero_shot_runner.py).
- Supervised Transformer baseline (supervised_runner.py).
- SVM + TF-IDF baseline (svm_runner.py).
- Sentence-BERT semantic similarity baseline (sbert_runner.py).

It then:
- Aggregates metrics across methods into a single comparison table.
- Creates simple bar plots comparing micro/macro F1, resource_macro_f1.
- Writes a text summary suitable as input to GPT for an experimental section.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import data_io


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: List[str]) -> None:
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_llm_methods(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run methods that require an OpenAI API key (runner + zero-shot + few-shot).

    Skips if OPENAI_API_KEY is not set or --skip_llm is True.
    """
    if args.skip_llm:
        print("\n[INFO] Skipping LLM-based methods because --skip_llm is set.")
        return

    # Accept key either from environment or from config.yaml (openai_api_key).
    api_key_env = os.environ.get("OPENAI_API_KEY")
    api_key_cfg = config.get("openai_api_key")

    if not (api_key_env or api_key_cfg):
        print("\n[WARN] No OpenAI API key found; skipping LLM-based methods (runner, zero_shot_runner).")
        print("       Set openai_api_key in config.yaml or export OPENAI_API_KEY in the shell before running.")
        return

    # Main BM25+LLM+graph pipeline + internal baselines
    run_cmd(
        [
            "python",
            "-m",
            "runner",
            "--data_dir",
            str(args.data_dir),
            "--gold_dir",
            str(args.gold_dir),
            "--competencies",
            str(args.competencies),
            "--config",
            str(args.config),
            "--output_dir",
            str(args.output_main),
        ]
    )

    # Zero-shot LLM baseline
    run_cmd(
        [
            "python",
            "zero_shot_runner.py",
            "--data_dir",
            str(args.data_dir),
            "--gold_dir",
            str(args.gold_dir),
            "--competencies",
            str(args.competencies),
            "--config",
            str(args.config),
            "--output_dir",
            str(args.output_zero_shot),
        ]
    )

    # Few-shot LLM baseline (in-context learning, no BM25, no graph)
    run_cmd(
        [
            "python",
            "few_shot_runner.py",
            "--data_dir",
            str(args.data_dir),
            "--gold_dir",
            str(args.gold_dir),
            "--competencies",
            str(args.competencies),
            "--config",
            str(args.config),
            "--output_dir",
            str(args.output_few_shot),
        ]
    )


def run_non_llm_methods(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run supervised Transformer, SVM, and SBERT baselines (no OpenAI key needed)."""
    # Supervised Transformer
    run_cmd(
        [
            "python",
            "supervised_runner.py",
            "--data_dir",
            str(args.data_dir),
            "--gold_dir",
            str(args.gold_dir),
            "--competencies",
            str(args.competencies),
            "--config",
            str(args.config),
            "--output_dir",
            str(args.output_supervised),
        ]
    )

    # SVM + TF-IDF
    run_cmd(
        [
            "python",
            "svm_runner.py",
            "--data_dir",
            str(args.data_dir),
            "--gold_dir",
            str(args.gold_dir),
            "--competencies",
            str(args.competencies),
            "--config",
            str(args.config),
            "--output_dir",
            str(args.output_svm),
        ]
    )

    # TF-IDF + OVR Logistic Regression
    if (Path(__file__).parent / "tfidf_logreg_runner.py").exists():
        run_cmd(
            [
                "python",
                "tfidf_logreg_runner.py",
                "--data_dir",
                str(args.data_dir),
                "--gold_dir",
                str(args.gold_dir),
                "--competencies",
                str(args.competencies),
                "--config",
                str(args.config),
                "--output_dir",
                str(args.output_logreg),
            ]
        )

    # TF-IDF + OVR LinearSVC (scores)
    if (Path(__file__).parent / "tfidf_linearsvc_runner.py").exists():
        run_cmd(
            [
                "python",
                "tfidf_linearsvc_runner.py",
                "--data_dir",
                str(args.data_dir),
                "--gold_dir",
                str(args.gold_dir),
                "--competencies",
                str(args.competencies),
                "--config",
                str(args.config),
                "--output_dir",
                str(args.output_linearsvc),
            ]
        )

    # SBERT semantic similarity
    run_cmd(
        [
            "python",
            "sbert_runner.py",
            "--data_dir",
            str(args.data_dir),
            "--gold_dir",
            str(args.gold_dir),
            "--competencies",
            str(args.competencies),
            "--config",
            str(args.config),
            "--output_dir",
            str(args.output_sbert),
        ]
    )


def safe_mean(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cols:
        if c in df.columns:
            out[c] = float(df[c].mean())
    return out


def aggregate_results(args: argparse.Namespace, config: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate metrics across all methods into a single table."""
    rows: List[Dict[str, Any]] = []

    metric_cols = [
        "micro_f1",
        "macro_f1",
        "resource_macro_f1",
        "micro_precision",
        "micro_recall",
        "evidence_valid_rate",
        "evidence_overlap_mean",
    ]

    # Main LLM+BM25+graph method (best config by micro_f1)
    metrics_global_path = args.output_main / "metrics_global.csv"
    if metrics_global_path.exists():
        mg = pd.read_csv(metrics_global_path)
        if "micro_f1" in mg.columns and not mg.empty:
            best_idx = mg["micro_f1"].idxmax()
            best = mg.iloc[best_idx]
            row = {
                "method": "LLM+BM25+Graph (best K,τ)",
                "retrieval_k": int(best["retrieval_k"]),
                "threshold": float(best["threshold"]),
            }
            for c in metric_cols:
                if c in best:
                    row[c] = float(best[c])
            rows.append(row)

    # Baselines inside runner: NONE, bm25_only, keyword
    baseline_path = args.output_main / "baseline_results.csv"
    if baseline_path.exists():
        bdf = pd.read_csv(baseline_path)
        for m in sorted(bdf["method"].unique()):
            sub = bdf[bdf["method"] == m]
            row = {"method": f"{m} (runner baseline)"}
            row.update(safe_mean(sub, metric_cols))
            rows.append(row)

    # Zero-shot LLM
    zs_path = args.output_zero_shot / "zero_shot_results.csv"
    if zs_path.exists():
        zs = pd.read_csv(zs_path)
        row = {"method": "Zero-shot LLM"}
        row.update(safe_mean(zs, metric_cols))
        rows.append(row)

    # Few-shot LLM
    fs_path = args.output_few_shot / "few_shot_results.csv"
    if fs_path.exists():
        fs = pd.read_csv(fs_path)
        row = {"method": "Few-shot LLM"}
        row.update(safe_mean(fs, metric_cols))
        rows.append(row)

    # Supervised Transformer
    sup_path = args.output_supervised / "supervised_results.csv"
    if sup_path.exists():
        sup = pd.read_csv(sup_path)
        row = {"method": "Supervised Transformer"}
        row.update(safe_mean(sup, metric_cols))
        rows.append(row)

    # SVM
    svm_path = args.output_svm / "svm_results.csv"
    if svm_path.exists():
        svm = pd.read_csv(svm_path)
        row = {"method": "SVM + TF-IDF"}
        row.update(safe_mean(svm, metric_cols))
        rows.append(row)

    # TF-IDF + OVR Logistic Regression
    logreg_path = args.output_logreg / "logreg_results.csv"
    if logreg_path.exists():
        lrdf = pd.read_csv(logreg_path)
        row = {"method": "TF-IDF + OVR LogReg"}
        row.update(safe_mean(lrdf, metric_cols))
        rows.append(row)

    # TF-IDF + OVR LinearSVC (scores)
    lsvc_path = args.output_linearsvc / "linearsvc_results.csv"
    if lsvc_path.exists():
        ldf = pd.read_csv(lsvc_path)
        row = {"method": "TF-IDF + OVR LinearSVC"}
        row.update(safe_mean(ldf, metric_cols))
        rows.append(row)

    # SBERT
    sbert_path = args.output_sbert / "sbert_results.csv"
    if sbert_path.exists():
        sbert = pd.read_csv(sbert_path)
        row = {"method": "SBERT similarity"}
        row.update(safe_mean(sbert, metric_cols))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_summary / "methods_comparison.csv", index=False)
    print("\n[INFO] Saved methods comparison to:", args.output_summary / "methods_comparison.csv")
    return df


def plot_method_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        print("[WARN] No methods to plot.")
        return

    sns.set(style="whitegrid")
    metrics_to_plot = ["micro_f1", "macro_f1", "resource_macro_f1"]

    plt.figure(figsize=(10, 6))
    df_plot = df[["method"] + [m for m in metrics_to_plot if m in df.columns]].set_index("method")
    df_plot = df_plot.sort_values(by="micro_f1", ascending=False)

    df_plot.plot(kind="bar")
    plt.ylabel("Score")
    plt.title("Method comparison (higher is better)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    png_path = out_dir / "methods_comparison.png"
    pdf_path = out_dir / "methods_comparison.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print("[INFO] Saved comparison plots to:", png_path, "and", pdf_path)


def compute_dataset_stats(data_dir: Path, gold_dir: Path, competencies_path: Path) -> Dict[str, Any]:
    comps = data_io.load_competencies(competencies_path)

    frag_files = data_io.find_fragment_files(data_dir)
    gold_files = data_io.find_gold_files(gold_dir)
    uv_to_frag = {data_io.get_uv_from_path(p): p for p in frag_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}
    uvs = sorted(set(uv_to_frag.keys()) & set(uv_to_gold.keys()))

    total_frags = 0
    total_gold = 0
    for uv in uvs:
        frags = data_io.load_fragments(uv_to_frag[uv])
        gold = data_io.load_gold_fragments(uv_to_gold[uv])
        total_frags += len(frags)
        total_gold += len(gold)

    return {
        "num_competencies": len(comps),
        "num_uvs": len(uvs),
        "total_fragments": total_frags,
        "total_gold_annotations": total_gold,
    }


def write_text_summary(
    args: argparse.Namespace,
    config: Dict[str, Any],
    dataset_stats: Dict[str, Any],
    methods_df: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("EXPERIMENTAL SETUP SUMMARY (for GPT prompt)\n")

    # Dataset
    lines.append("Dataset and task")
    lines.append("- Number of competencies: {}".format(dataset_stats["num_competencies"]))
    lines.append("- Number of UVs (course units): {}".format(dataset_stats["num_uvs"]))
    lines.append("- Total fragments: {}".format(dataset_stats["total_fragments"]))
    lines.append("- Total fragment-level gold annotations: {}".format(dataset_stats["total_gold_annotations"]))
    lines.append("")

    # CV + sweeps
    lines.append("Cross-validation and hyperparameters")
    lines.append("- Cross-validation: {}-fold, splitting by UV".format(config.get("n_folds", 5)))
    lines.append(
        "- Retrieval K sweep (main pipeline): {}".format(config.get("sweep_retrieval_k", []))
    )
    lines.append(
        "- Decision threshold sweep (main pipeline): {}".format(
            config.get("sweep_threshold", [])
        )
    )
    lines.append("- LLM model: {}".format(config.get("llm_model", "gpt-4o-mini")))
    lines.append("")

    # Methods
    lines.append("Compared methods")
    lines.append("- LLM+BM25+Graph: BM25 candidate retrieval, LLM tagging with evidence spans, graph-based reconciliation, resource-level aggregation.")
    lines.append("- Baselines inside runner: NONE (always predicts no label), BM25-only (top-k candidates), keyword matching (string overlap with competency terms).")
    lines.append("- Zero-shot LLM: prompts the LLM with all competencies, without BM25 or graph, to assign labels directly.")
    lines.append("- Few-shot LLM: similar to zero-shot but with a small set of labeled fragment examples provided in the prompt as in-context supervision (no retrieval, no graph).")
    lines.append("- Supervised Transformer: multi-label sequence classifier fine-tuned on fragments with gold competencies.")
    lines.append("- SVM + TF-IDF: one-vs-rest linear SVM trained on TF-IDF features of fragment text.")
    lines.append("- SBERT similarity: pre-trained sentence-transformer to embed fragments and competencies, cosine similarity to rank competencies.")
    lines.append("")

    # Metrics table
    lines.append("Average performance across methods (values are means over UVs/folds):")
    if not methods_df.empty:
        cols_for_text = [
            "method",
            "micro_f1",
            "macro_f1",
            "resource_macro_f1",
            "evidence_valid_rate",
        ]
        sub = methods_df[[c for c in cols_for_text if c in methods_df.columns]].copy()
        lines.append(sub.to_csv(index=False))
    else:
        lines.append("(no methods_df data available)")

    lines.append("")
    lines.append("Guidance for writing the experimental section:")
    lines.append("- Describe the dataset using the numbers above (competencies, UVs, fragments).")
    lines.append("- Explain that the main method combines BM25 retrieval, LLM tagging with evidence, and graph-based reconciliation, then is compared against traditional IR, supervised classifiers (Transformer, SVM), semantic similarity (SBERT), and zero-shot LLM baselines.")
    lines.append("- Emphasize that evaluation is multi-label at fragment and resource levels, using micro/macro F1, and evidence validity/overlap for methods that output spans.")
    lines.append("- Highlight which method achieves the best micro_F1 and macro_F1, and how far above simple baselines it is.")

    out_path = args.output_summary / "experiments_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[INFO] Wrote text summary to:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all competency-tagging experiments and summarize results")
    parser.add_argument("--data_dir", type=Path, default=Path("resources_fragments"))
    parser.add_argument("--gold_dir", type=Path, default=Path("golds_fragments"))
    parser.add_argument("--competencies", type=Path, default=Path("competencies_utc.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))

    parser.add_argument("--output_main", type=Path, default=Path("output"))
    parser.add_argument("--output_zero_shot", type=Path, default=Path("output_zero_shot"))
    parser.add_argument("--output_few_shot", type=Path, default=Path("output_few_shot"))
    parser.add_argument("--output_supervised", type=Path, default=Path("output_supervised"))
    parser.add_argument("--output_svm", type=Path, default=Path("output_svm"))
    parser.add_argument("--output_logreg", type=Path, default=Path("output_logreg"))
    parser.add_argument("--output_linearsvc", type=Path, default=Path("output_linearsvc"))
    parser.add_argument("--output_sbert", type=Path, default=Path("output_sbert"))
    parser.add_argument("--output_summary", type=Path, default=Path("output_summary"))

    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Skip LLM-based methods (runner + zero_shot_runner)",
    )

    args = parser.parse_args()

    ensure_dir(args.output_main)
    ensure_dir(args.output_zero_shot)
    ensure_dir(args.output_few_shot)
    ensure_dir(args.output_supervised)
    ensure_dir(args.output_svm)
    ensure_dir(args.output_logreg)
    ensure_dir(args.output_linearsvc)
    ensure_dir(args.output_sbert)
    ensure_dir(args.output_summary)

    config = load_config(args.config)

    # 1) Run methods
    run_llm_methods(args, config)
    run_non_llm_methods(args, config)

    # 2) Aggregate and plot
    methods_df = aggregate_results(args, config)
    plot_method_comparison(methods_df, args.output_summary)

    # 3) Dataset stats and text summary for GPT
    dataset_stats = compute_dataset_stats(args.data_dir, args.gold_dir, args.competencies)
    write_text_summary(args, config, dataset_stats, methods_df)


if __name__ == "__main__":
    main()

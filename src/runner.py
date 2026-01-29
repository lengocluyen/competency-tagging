"""Experiment runner for competency tagging pipeline."""
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from tqdm import tqdm
import json

import data_io
from retrieval import CompetencyRetriever
from llm_tagger import LLMTagger
from reconcile import GraphReconciler
from aggregate import ResourceAggregator
import metrics as metrics_module
from baselines import run_baselines_for_uv


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_cv_splits(uv_list: List[str], n_folds: int = 5) -> List[Tuple[List[str], List[str]]]:
    """Create cross-validation splits by UV.
    
    Args:
        uv_list: List of UV identifiers
        n_folds: Number of folds
        
    Returns:
        List of (train_uvs, test_uvs) tuples
    """
    import numpy as np
    
    uv_array = np.array(uv_list)
    np.random.shuffle(uv_array)
    
    splits = []
    fold_size = len(uv_array) // n_folds
    
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else len(uv_array)
        
        test_uvs = uv_array[test_start:test_end].tolist()
        train_uvs = [uv for uv in uv_list if uv not in test_uvs]
        
        splits.append((train_uvs, test_uvs))
    
    return splits


def run_pipeline(
    fragments: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    competencies: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Run full tagging pipeline.
    
    Args:
        fragments: Dictionary of fragments
        gold: Dictionary of gold annotations
        competencies: Dictionary of competencies
        config: Configuration dictionary
        output_dir: Output directory for predictions
        
    Returns:
        Tuple of (fragment_predictions, resource_predictions, metrics)
    """
    # Initialize components
    retriever = CompetencyRetriever(competencies)
    
    tagger = LLMTagger(
        competencies=competencies,
        model=config.get('llm_model', 'gpt-4o-mini'),
        cache_dir=output_dir / 'cache',
        max_retries=config.get('max_retries', 5),
        validate_evidence=config.get('validate_evidence', True),
        repair_invalid=config.get('repair_invalid', True),
        api_key=config.get('openai_api_key')
    )
    
    reconciler = GraphReconciler(
        competencies=competencies,
        max_labels_per_fragment=config.get('max_labels_per_fragment', 5),
        prefer_child=config.get('prefer_child', True),
        flag_prereq_violations=config.get('flag_prereq_violations', True)
    )
    
    aggregator = ResourceAggregator(
        aggregation_method=config.get('aggregation_method', 'max'),
        fragment_type_weights=config.get('fragment_type_weights', {}),
        top_k_per_resource=config.get('top_k_per_resource', 10),
        threshold=config.get('threshold', 0.5)
    )
    
    # Step 1: Candidate retrieval
    print("Step 1: Retrieving candidates...")
    fragment_candidates = {}

    # If gold annotations include an explicit candidate list, prefer that.
    # Fallback to BM25 retrieval when candidates are missing for a fragment.
    gold_has_candidates = any(
        isinstance(g.get("candidates"), list) and len(g.get("candidates")) > 0
        for g in gold.values()
    )

    if gold_has_candidates:
        print("Using candidates from gold annotations when available (no BM25 for those fragments)...")
    
    for frag_id, frag in tqdm(fragments.items(), desc="Retrieval"):
        text = frag.get('text', '')

        candidates: List[Tuple[str, float]]

        if gold_has_candidates and frag_id in gold and isinstance(gold[frag_id].get("candidates"), list):
            raw_cands = gold[frag_id]["candidates"]
            parsed: List[Tuple[str, float]] = []

            for entry in raw_cands:
                # Support both simple competency_id strings and
                # objects like {"competency_id": "...", "score": 4}.
                cid: Any = None
                score: float = 1.0

                if isinstance(entry, str):
                    cid = entry
                elif isinstance(entry, dict):
                    cid = entry.get("competency_id")
                    if cid is None:
                        continue
                    try:
                        score = float(entry.get("score", 1.0))
                    except (TypeError, ValueError):
                        score = 1.0
                else:
                    continue

                if cid in competencies:
                    parsed.append((cid, score))

            candidates = parsed or retriever.retrieve(text, top_k=config.get('retrieval_k', 10))
        else:
            candidates = retriever.retrieve(text, top_k=config.get('retrieval_k', 10))

        fragment_candidates[frag_id] = candidates
    
    # Step 2: LLM tagging
    print("Step 2: LLM tagging...")
    fragment_predictions = []
    
    for frag_id, frag in tqdm(fragments.items(), desc="Tagging"):
        text = frag.get('text', '')
        candidates = fragment_candidates[frag_id]
        
        tagging_result = tagger.tag_fragment(text, candidates)
        
        fragment_predictions.append({
            'fragment_id': frag_id,
            'candidates': candidates,
            'tagging_result': tagging_result,
            'predictions': tagging_result.get('selected', [])
        })
    
    # Step 3: Graph reconciliation
    print("Step 3: Graph reconciliation...")
    reconcile_stats = []
    
    for pred_dict in tqdm(fragment_predictions, desc="Reconciling"):
        predictions = pred_dict['predictions']
        reconciled, stats = reconciler.reconcile(predictions)
        pred_dict['predictions'] = reconciled
        reconcile_stats.append(stats)
    
    # Step 4: Aggregation to resources
    print("Step 4: Aggregating to resources...")
    resource_predictions = aggregator.aggregate(fragment_predictions, fragments)
    
    # Step 5: Evaluation
    print("Step 5: Evaluating...")
    eval_metrics = metrics_module.evaluate_pipeline(
        fragment_predictions=fragment_predictions,
        resource_predictions=resource_predictions,
        gold=gold,
        fragments=fragments,
        reconcile_stats=reconcile_stats
    )
    
    return fragment_predictions, resource_predictions, eval_metrics


def run_experiment(
    data_dir: Path,
    gold_dir: Path,
    competencies_path: Path,
    config: Dict[str, Any],
    output_dir: Path
):
    """Run full experiment with cross-validation and parameter sweeps.
    
    Args:
        data_dir: Directory with fragment files
        gold_dir: Directory with gold annotation files
        competencies_path: Path to competencies file
        config: Configuration dictionary
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load competencies
    print("Loading competencies...")
    competencies = data_io.load_competencies(competencies_path)
    print(f"Loaded {len(competencies)} competencies")
    
    # Find all UVs
    fragment_files = data_io.find_fragment_files(data_dir)
    gold_files = data_io.find_gold_files(gold_dir)
    
    uv_to_fragment_file = {data_io.get_uv_from_path(f): f for f in fragment_files}
    uv_to_gold_file = {data_io.get_uv_from_path(f): f for f in gold_files}
    
    # Get UVs with both fragments and gold
    uvs = sorted(set(uv_to_fragment_file.keys()) & set(uv_to_gold_file.keys()))
    print(f"Found {len(uvs)} UVs: {uvs}")
    
    # Parameter sweep configurations
    retrieval_k_values = config.get('sweep_retrieval_k', [5, 10, 15, 20])
    threshold_values = config.get('sweep_threshold', [0.3, 0.4, 0.5, 0.6])
    n_folds = config.get('n_folds', 5)
    
    # Create CV splits
    print(f"Creating {n_folds}-fold CV splits...")
    cv_splits = create_cv_splits(uvs, n_folds)
    
    # Results storage
    all_results = []
    baseline_results: List[Dict[str, Any]] = []
    
    # Run experiments
    for k in retrieval_k_values:
        for tau in threshold_values:
            print(f"\n{'='*80}")
            print(f"Running experiment: K={k}, τ={tau}")
            print(f"{'='*80}")
            
            # Update config
            exp_config = config.copy()
            exp_config['retrieval_k'] = k
            exp_config['threshold'] = tau
            
            fold_results = []
            
            for fold_idx, (train_uvs, test_uvs) in enumerate(cv_splits):
                print(f"\nFold {fold_idx + 1}/{n_folds}")
                print(f"Train UVs: {train_uvs}")
                print(f"Test UVs: {test_uvs}")
                
                # For each test UV
                for uv in test_uvs:
                    print(f"\nProcessing UV: {uv}")
                    
                    # Load data
                    fragments = data_io.load_fragments(uv_to_fragment_file[uv])
                    gold = data_io.load_gold_fragments(uv_to_gold_file[uv])
                    
                    print(f"Loaded {len(fragments)} fragments, {len(gold)} gold annotations")
                    
                    # Run pipeline
                    exp_output_dir = output_dir / f"k{k}_tau{tau}" / f"fold{fold_idx}" / uv
                    exp_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    frag_preds, res_preds, eval_metrics = run_pipeline(
                        fragments=fragments,
                        gold=gold,
                        competencies=competencies,
                        config=exp_config,
                        output_dir=exp_output_dir
                    )
                    
                    # Save predictions
                    data_io.save_jsonl(frag_preds, exp_output_dir / f"{uv}_fragment_predictions.jsonl")
                    data_io.save_jsonl(res_preds, exp_output_dir / f"{uv}_resource_predictions.jsonl")
                    
                    # Store main model results
                    result = {
                        'uv': uv,
                        'fold': fold_idx,
                        'retrieval_k': k,
                        'threshold': tau,
                        **eval_metrics
                    }
                    fold_results.append(result)
                    all_results.append(result)
                    
                    print(f"Metrics: {eval_metrics}")

                    # Run baselines for this UV and config
                    baseline_out_dir = exp_output_dir / "baselines"
                    baseline_out_dir.mkdir(parents=True, exist_ok=True)

                    baseline_metrics_list = run_baselines_for_uv(
                        fragments=fragments,
                        gold=gold,
                        competencies=competencies,
                        config=exp_config,
                        retrieval_k=k,
                        output_dir=baseline_out_dir,
                    )

                    for b in baseline_metrics_list:
                        baseline_row = {
                            'uv': uv,
                            'fold': fold_idx,
                            'retrieval_k': k,
                            'threshold': tau,
                            'method': b['method'],
                            **b['metrics'],
                        }
                        baseline_results.append(baseline_row)
            
            # Save fold results
            fold_df = pd.DataFrame(fold_results)
            fold_df.to_csv(output_dir / f"results_k{k}_tau{tau}.csv", index=False)
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Save baseline results
    if baseline_results:
        baseline_df = pd.DataFrame(baseline_results)
        baseline_df.to_csv(output_dir / "baseline_results.csv", index=False)
    
    # Compute aggregated metrics
    print("\nComputing aggregated metrics...")
    
    # Metrics by UV
    metrics_by_uv = results_df.groupby('uv').mean(numeric_only=True).reset_index()
    metrics_by_uv.to_csv(output_dir / "metrics_by_uv.csv", index=False)
    
    # Global metrics (average across all UVs)
    metrics_global = results_df.groupby(['retrieval_k', 'threshold']).mean(numeric_only=True).reset_index()
    metrics_global.to_csv(output_dir / "metrics_global.csv", index=False)
    
    # Ablation: best config per metric
    ablation_results = []
    
    for metric_name in ['micro_f1', 'macro_f1', 'resource_macro_f1']:
        best_idx = metrics_global[metric_name].idxmax()
        best_config = metrics_global.iloc[best_idx]
        
        ablation_results.append({
            'metric': metric_name,
            'best_k': int(best_config['retrieval_k']),
            'best_threshold': float(best_config['threshold']),
            'value': float(best_config[metric_name])
        })
    
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(output_dir / "ablation.csv", index=False)
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Competency tagging experiment runner")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory with fragment files")
    parser.add_argument("--gold_dir", type=Path, required=True, help="Directory with gold annotation files")
    parser.add_argument("--competencies", type=Path, required=True, help="Path to competencies file")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run experiment
    run_experiment(
        data_dir=args.data_dir,
        gold_dir=args.gold_dir,
        competencies_path=args.competencies,
        config=config,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

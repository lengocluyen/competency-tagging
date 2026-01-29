"""Evaluation metrics for competency tagging."""
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import numpy as np


def compute_fragment_metrics(
    predictions: List[Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Compute fragment-level metrics.
    
    Args:
        predictions: List of fragment predictions with fragment_id, predictions
        gold: Dictionary of fragment_id -> gold annotations
        
    Returns:
        Dictionary of metrics
    """
    # Collect all predictions and gold labels
    all_pred_pairs = []  # (fragment_id, competency_id)
    all_gold_pairs = []
    
    fragment_ids = set()
    
    for pred_dict in predictions:
        frag_id = pred_dict['fragment_id']
        fragment_ids.add(frag_id)
        
        preds = pred_dict.get('predictions', [])
        for p in preds:
            all_pred_pairs.append((frag_id, p['competency_id']))
    
    for frag_id, gold_dict in gold.items():
        fragment_ids.add(frag_id)
        
        gold_labels = gold_dict.get('gold', [])
        for g in gold_labels:
            all_gold_pairs.append((frag_id, g['competency_id']))
    
    # Convert to sets for comparison
    pred_set = set(all_pred_pairs)
    gold_set = set(all_gold_pairs)
    
    # Micro metrics
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Macro F1 (per fragment)
    fragment_f1s = []
    
    for frag_id in fragment_ids:
        pred_comps = {c for (f, c) in pred_set if f == frag_id}
        gold_comps = {c for (f, c) in gold_set if f == frag_id}
        
        frag_tp = len(pred_comps & gold_comps)
        frag_fp = len(pred_comps - gold_comps)
        frag_fn = len(gold_comps - pred_comps)
        
        frag_p = frag_tp / (frag_tp + frag_fp) if (frag_tp + frag_fp) > 0 else 0.0
        frag_r = frag_tp / (frag_tp + frag_fn) if (frag_tp + frag_fn) > 0 else 0.0
        frag_f1 = 2 * frag_p * frag_r / (frag_p + frag_r) if (frag_p + frag_r) > 0 else 0.0
        
        fragment_f1s.append(frag_f1)
    
    macro_f1 = np.mean(fragment_f1s) if fragment_f1s else 0.0
    
    # NONE accuracy
    none_correct = 0
    none_total = 0
    
    for frag_id in fragment_ids:
        pred_comps = {c for (f, c) in pred_set if f == frag_id}
        gold_comps = {c for (f, c) in gold_set if f == frag_id}
        
        gold_none = frag_id in gold and gold[frag_id].get('gold_none', False)
        pred_none = len(pred_comps) == 0
        
        if gold_none or pred_none:
            none_total += 1
            if gold_none == pred_none:
                none_correct += 1
    
    none_accuracy = none_correct / none_total if none_total > 0 else 0.0
    
    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'none_accuracy': none_accuracy
    }


def compute_evidence_metrics(
    predictions: List[Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    fragments: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Compute evidence-related metrics.
    
    Args:
        predictions: List of fragment predictions
        gold: Dictionary of gold annotations
        fragments: Dictionary of fragment data
        
    Returns:
        Dictionary of evidence metrics
    """
    evidence_valid_count = 0
    evidence_total = 0
    evidence_overlap_scores = []
    
    for pred_dict in predictions:
        frag_id = pred_dict['fragment_id']
        preds = pred_dict.get('predictions', [])
        
        if frag_id not in gold or frag_id not in fragments:
            continue
        
        gold_dict = gold[frag_id]
        fragment_text = fragments[frag_id].get('text', '')
        
        # Build gold evidence map
        gold_evidence_map = {}
        for g in gold_dict.get('gold', []):
            comp_id = g['competency_id']
            evidence = g.get('evidence', {})
            gold_evidence_map[comp_id] = evidence
        
        for p in preds:
            comp_id = p['competency_id']
            pred_evidence = p.get('evidence', {})
            
            if not pred_evidence or 'quote' not in pred_evidence:
                continue
            
            evidence_total += 1
            
            # Check if prediction is correct
            gold_labels = {g['competency_id'] for g in gold_dict.get('gold', [])}
            if comp_id not in gold_labels:
                continue
            
            # Prediction is correct, check evidence
            evidence_valid_count += 1
            
            # Compute overlap with gold evidence
            if comp_id in gold_evidence_map:
                gold_ev = gold_evidence_map[comp_id]
                
                if 'start' in gold_ev and 'end' in gold_ev:
                    gold_start = gold_ev['start']
                    gold_end = gold_ev['end']
                    
                    pred_start = pred_evidence.get('start_char', 0)
                    pred_end = pred_evidence.get('end_char', 0)
                    
                    # Compute character IoU
                    overlap_start = max(gold_start, pred_start)
                    overlap_end = min(gold_end, pred_end)
                    overlap_len = max(0, overlap_end - overlap_start)
                    
                    union_start = min(gold_start, pred_start)
                    union_end = max(gold_end, pred_end)
                    union_len = max(1, union_end - union_start)
                    
                    iou = overlap_len / union_len
                    evidence_overlap_scores.append(iou)
    
    evidence_valid_rate = evidence_valid_count / evidence_total if evidence_total > 0 else 0.0
    evidence_overlap_mean = np.mean(evidence_overlap_scores) if evidence_overlap_scores else 0.0
    
    return {
        'evidence_valid_rate': evidence_valid_rate,
        'evidence_overlap_mean': evidence_overlap_mean
    }


def compute_coherence_metrics(
    predictions: List[Dict[str, Any]],
    reconcile_stats: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute coherence metrics from reconciliation stats.
    
    Args:
        predictions: List of fragment predictions
        reconcile_stats: List of reconciliation statistics
        
    Returns:
        Dictionary of coherence metrics
    """
    total_fragments = len(reconcile_stats)
    
    parent_child_redundancy_count = sum(
        s.get('parent_child_redundancy', 0) for s in reconcile_stats
    )
    
    prereq_violation_count = sum(
        s.get('prereq_violations', 0) for s in reconcile_stats
    )
    
    parent_child_redundancy_rate = parent_child_redundancy_count / total_fragments if total_fragments > 0 else 0.0
    prereq_violation_rate = prereq_violation_count / total_fragments if total_fragments > 0 else 0.0
    
    return {
        'parent_child_redundancy_rate': parent_child_redundancy_rate,
        'prereq_violation_rate': prereq_violation_rate
    }


def compute_resource_metrics(
    resource_predictions: List[Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    fragments: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Compute resource-level metrics.
    
    Args:
        resource_predictions: List of resource-level predictions
        gold: Dictionary of fragment-level gold annotations
        fragments: Dictionary of fragment data
        
    Returns:
        Dictionary of resource metrics
    """
    # Build gold resource labels from fragment gold
    gold_resource_labels: Dict[str, Set[str]] = defaultdict(set)
    
    for frag_id, gold_dict in gold.items():
        if frag_id not in fragments:
            continue
        
        resource_id = fragments[frag_id].get('resource_id')
        if not resource_id:
            continue
        
        for g in gold_dict.get('gold', []):
            gold_resource_labels[resource_id].add(g['competency_id'])
    
    # Collect predictions
    pred_resource_labels: Dict[str, Set[str]] = {}
    
    for res_pred in resource_predictions:
        resource_id = res_pred['resource_id']
        pred_comps = {p['competency_id'] for p in res_pred.get('predictions', [])}
        pred_resource_labels[resource_id] = pred_comps
    
    # Compute metrics
    all_resources = set(gold_resource_labels.keys()) | set(pred_resource_labels.keys())
    
    resource_f1s = []
    
    for resource_id in all_resources:
        pred_comps = pred_resource_labels.get(resource_id, set())
        gold_comps = gold_resource_labels.get(resource_id, set())
        
        tp = len(pred_comps & gold_comps)
        fp = len(pred_comps - gold_comps)
        fn = len(gold_comps - pred_comps)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        resource_f1s.append(f1)
    
    resource_macro_f1 = np.mean(resource_f1s) if resource_f1s else 0.0
    
    # Micro metrics
    all_pred_pairs = [(r, c) for r, comps in pred_resource_labels.items() for c in comps]
    all_gold_pairs = [(r, c) for r, comps in gold_resource_labels.items() for c in comps]
    
    pred_set = set(all_pred_pairs)
    gold_set = set(all_gold_pairs)
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    return {
        'resource_macro_f1': resource_macro_f1,
        'resource_micro_f1': micro_f1,
        'resource_micro_precision': micro_p,
        'resource_micro_recall': micro_r
    }


def evaluate_pipeline(
    fragment_predictions: List[Dict[str, Any]],
    resource_predictions: List[Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    fragments: Dict[str, Dict[str, Any]],
    reconcile_stats: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Evaluate full pipeline with all metrics.
    
    Args:
        fragment_predictions: Fragment-level predictions
        resource_predictions: Resource-level predictions
        gold: Gold annotations
        fragments: Fragment data
        reconcile_stats: Reconciliation statistics
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Fragment-level metrics
    frag_metrics = compute_fragment_metrics(fragment_predictions, gold)
    metrics.update(frag_metrics)
    
    # Evidence metrics
    ev_metrics = compute_evidence_metrics(fragment_predictions, gold, fragments)
    metrics.update(ev_metrics)
    
    # Coherence metrics
    coh_metrics = compute_coherence_metrics(fragment_predictions, reconcile_stats)
    metrics.update(coh_metrics)
    
    # Resource-level metrics
    res_metrics = compute_resource_metrics(resource_predictions, gold, fragments)
    metrics.update(res_metrics)
    
    return metrics

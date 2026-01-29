"""Aggregate fragment-level predictions to resource-level tags."""
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class ResourceAggregator:
    """Aggregate fragment predictions to resource-level tags."""
    
    def __init__(
        self,
        aggregation_method: str = "max",
        fragment_type_weights: Dict[str, float] = None,
        top_k_per_resource: int = 10,
        threshold: float = 0.5
    ):
        """Initialize aggregator.
        
        Args:
            aggregation_method: Method for aggregation ('max' or 'weighted_sum')
            fragment_type_weights: Weights for different fragment types
            top_k_per_resource: Number of top tags to keep per resource
            threshold: Minimum confidence threshold for tags
        """
        self.aggregation_method = aggregation_method
        self.fragment_type_weights = fragment_type_weights or {}
        self.top_k_per_resource = top_k_per_resource
        self.threshold = threshold
    
    def aggregate(
        self,
        fragment_predictions: List[Dict[str, Any]],
        fragments: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aggregate fragment predictions to resource level.
        
        Args:
            fragment_predictions: List of fragment predictions with fragment_id, predictions
            fragments: Dictionary of fragment_id -> fragment data
            
        Returns:
            List of resource-level predictions
        """
        # Group by resource
        resource_to_fragments: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]] = defaultdict(list)
        
        for frag_pred in fragment_predictions:
            frag_id = frag_pred['fragment_id']
            predictions = frag_pred.get('predictions', [])
            
            if frag_id not in fragments:
                continue
            
            resource_id = fragments[frag_id].get('resource_id')
            if resource_id:
                resource_to_fragments[resource_id].append((frag_id, predictions))
        
        # Aggregate each resource
        resource_predictions = []
        
        for resource_id, frag_data in resource_to_fragments.items():
            # Aggregate competency scores
            comp_scores: Dict[str, List[float]] = defaultdict(list)
            comp_fragments: Dict[str, List[str]] = defaultdict(list)
            
            for frag_id, predictions in frag_data:
                fragment = fragments[frag_id]
                fragment_type = fragment.get('fragment_type', 'default')
                weight = self.fragment_type_weights.get(fragment_type, 1.0)
                
                for pred in predictions:
                    comp_id = pred['competency_id']
                    confidence = pred.get('confidence', 0.0)
                    
                    # Apply fragment type weight
                    weighted_conf = confidence * weight
                    
                    comp_scores[comp_id].append(weighted_conf)
                    comp_fragments[comp_id].append(frag_id)
            
            # Compute final scores
            comp_final_scores = {}
            
            for comp_id, scores in comp_scores.items():
                if self.aggregation_method == "max":
                    final_score = max(scores)
                elif self.aggregation_method == "weighted_sum":
                    final_score = sum(scores) / len(scores)  # Average of weighted scores
                else:
                    final_score = max(scores)
                
                comp_final_scores[comp_id] = final_score
            
            # Filter by threshold and select top-K
            filtered = [
                (comp_id, score)
                for comp_id, score in comp_final_scores.items()
                if score >= self.threshold
            ]
            
            # Sort by score descending
            filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
            
            # Keep top-K
            top_k = filtered[:self.top_k_per_resource]
            
            # Build resource prediction
            resource_pred = {
                'resource_id': resource_id,
                'predictions': [
                    {
                        'competency_id': comp_id,
                        'confidence': score,
                        'supporting_fragments': comp_fragments[comp_id]
                    }
                    for comp_id, score in top_k
                ]
            }
            
            resource_predictions.append(resource_pred)
        
        return resource_predictions
    
    def aggregate_batch(
        self,
        batch_fragment_predictions: List[List[Dict[str, Any]]],
        fragments_list: List[Dict[str, Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """Aggregate batch of fragment predictions.
        
        Args:
            batch_fragment_predictions: List of fragment prediction lists
            fragments_list: List of fragment dictionaries
            
        Returns:
            List of resource prediction lists
        """
        results = []
        
        for frag_preds, fragments in zip(batch_fragment_predictions, fragments_list):
            resource_preds = self.aggregate(frag_preds, fragments)
            results.append(resource_preds)
        
        return results

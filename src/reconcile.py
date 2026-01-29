"""Graph-based reconciliation using competency relationships."""
from typing import Dict, List, Any, Set, Tuple


class GraphReconciler:
    """Reconcile predictions using competency graph structure."""
    
    def __init__(
        self,
        competencies: Dict[str, Dict[str, Any]],
        max_labels_per_fragment: int = 5,
        prefer_child: bool = True,
        flag_prereq_violations: bool = True
    ):
        """Initialize reconciler.
        
        Args:
            competencies: Dictionary of competency_id -> competency record
            max_labels_per_fragment: Maximum number of labels to keep per fragment
            prefer_child: If parent and child predicted, keep child (unless only parent has evidence)
            flag_prereq_violations: Flag prerequisite violations
        """
        self.competencies = competencies
        self.max_labels_per_fragment = max_labels_per_fragment
        self.prefer_child = prefer_child
        self.flag_prereq_violations = flag_prereq_violations
        
        # Build relationship indices
        self._build_indices()
    
    def _build_indices(self) -> None:
        """Build parent-child and prerequisite indices."""
        self.parent_to_children: Dict[str, Set[str]] = {}
        self.child_to_parents: Dict[str, Set[str]] = {}
        self.comp_to_prereqs: Dict[str, Set[str]] = {}
        
        for comp_id, comp in self.competencies.items():
            neighbors = comp.get('neighbors', {})
            
            # Process parent relationships
            if 'parent' in neighbors:
                parents = neighbors['parent']
                if not isinstance(parents, list):
                    parents = [parents]
                
                for parent in parents:
                    if parent not in self.parent_to_children:
                        self.parent_to_children[parent] = set()
                    self.parent_to_children[parent].add(comp_id)
                    
                    if comp_id not in self.child_to_parents:
                        self.child_to_parents[comp_id] = set()
                    self.child_to_parents[comp_id].add(parent)
            
            # Process children relationships
            if 'children' in neighbors:
                children = neighbors['children']
                if not isinstance(children, list):
                    children = [children]
                
                for child in children:
                    if comp_id not in self.parent_to_children:
                        self.parent_to_children[comp_id] = set()
                    self.parent_to_children[comp_id].add(child)
                    
                    if child not in self.child_to_parents:
                        self.child_to_parents[child] = set()
                    self.child_to_parents[child].add(comp_id)
            
            # Process prerequisites
            if 'prereq' in neighbors:
                prereqs = neighbors['prereq']
                if not isinstance(prereqs, list):
                    prereqs = [prereqs]
                
                self.comp_to_prereqs[comp_id] = set(prereqs)
    
    def _find_parent_child_pairs(self, comp_ids: List[str]) -> List[Tuple[str, str]]:
        """Find parent-child pairs in predicted competencies.
        
        Args:
            comp_ids: List of predicted competency IDs
            
        Returns:
            List of (parent, child) tuples
        """
        comp_set = set(comp_ids)
        pairs = []
        
        for comp_id in comp_ids:
            # Check if any of its parents are in predictions
            parents = self.child_to_parents.get(comp_id, set())
            for parent in parents:
                if parent in comp_set:
                    pairs.append((parent, comp_id))
        
        return pairs
    
    def _has_valid_evidence(self, prediction: Dict[str, Any]) -> bool:
        """Check if prediction has valid evidence.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            True if has valid evidence
        """
        evidence = prediction.get('evidence')
        if not evidence:
            return False
        
        return 'quote' in evidence and evidence['quote']
    
    def reconcile(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Reconcile predictions using graph rules.
        
        Args:
            predictions: List of predictions with competency_id, confidence, evidence
            
        Returns:
            Tuple of (reconciled_predictions, stats_dict)
        """
        if not predictions:
            return [], {}
        
        reconciled = list(predictions)
        stats = {
            'parent_child_redundancy': 0,
            'prereq_violations': 0,
            'capped_by_max_labels': 0
        }
        
        # Get competency IDs
        comp_ids = [p['competency_id'] for p in reconciled]
        
        # Rule 1: Resolve parent-child redundancy
        if self.prefer_child:
            pairs = self._find_parent_child_pairs(comp_ids)
            to_remove = set()
            
            for parent, child in pairs:
                # Find predictions
                parent_pred = next((p for p in reconciled if p['competency_id'] == parent), None)
                child_pred = next((p for p in reconciled if p['competency_id'] == child), None)
                
                if not parent_pred or not child_pred:
                    continue
                
                # Keep child unless only parent has valid evidence
                parent_has_evidence = self._has_valid_evidence(parent_pred)
                child_has_evidence = self._has_valid_evidence(child_pred)
                
                if child_has_evidence or not parent_has_evidence:
                    # Remove parent
                    to_remove.add(parent)
                    stats['parent_child_redundancy'] += 1
                else:
                    # Remove child (only parent has evidence)
                    to_remove.add(child)
                    stats['parent_child_redundancy'] += 1
            
            reconciled = [p for p in reconciled if p['competency_id'] not in to_remove]
        
        # Rule 2: Check prerequisite violations
        if self.flag_prereq_violations:
            comp_ids = {p['competency_id'] for p in reconciled}
            violations = []
            
            for comp_id in comp_ids:
                prereqs = self.comp_to_prereqs.get(comp_id, set())
                missing_prereqs = prereqs - comp_ids
                
                if missing_prereqs:
                    violations.append({
                        'competency_id': comp_id,
                        'missing_prereqs': list(missing_prereqs)
                    })
                    stats['prereq_violations'] += 1
            
            # Add violations to each prediction
            for pred in reconciled:
                comp_id = pred['competency_id']
                prereqs = self.comp_to_prereqs.get(comp_id, set())
                missing_prereqs = prereqs - comp_ids
                
                if missing_prereqs:
                    pred['prereq_violation'] = True
                    pred['missing_prereqs'] = list(missing_prereqs)
        
        # Rule 3: Cap max labels per fragment
        if len(reconciled) > self.max_labels_per_fragment:
            # Sort by confidence and keep top-K
            reconciled = sorted(reconciled, key=lambda x: x.get('confidence', 0), reverse=True)
            reconciled = reconciled[:self.max_labels_per_fragment]
            stats['capped_by_max_labels'] = 1
        
        return reconciled, stats
    
    def reconcile_batch(
        self,
        batch_predictions: List[List[Dict[str, Any]]]
    ) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Reconcile batch of predictions.
        
        Args:
            batch_predictions: List of prediction lists
            
        Returns:
            Tuple of (reconciled_predictions_list, stats_list)
        """
        reconciled_list = []
        stats_list = []
        
        for predictions in batch_predictions:
            reconciled, stats = self.reconcile(predictions)
            reconciled_list.append(reconciled)
            stats_list.append(stats)
        
        return reconciled_list, stats_list

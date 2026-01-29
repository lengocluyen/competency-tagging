"""BM25-based candidate retrieval for competency tagging."""
from typing import Dict, List, Tuple, Any
from rank_bm25 import BM25Okapi
import data_io


class CompetencyRetriever:
    """BM25-based retriever for competency candidates."""
    
    def __init__(self, competencies: Dict[str, Dict[str, Any]]):
        """Initialize retriever with competencies.
        
        Args:
            competencies: Dictionary of competency_id -> competency record
        """
        self.competencies = competencies
        self.comp_ids = list(competencies.keys())
        
        # Build profiles and tokenize
        self.profiles = []
        for comp_id in self.comp_ids:
            profile_text = data_io.build_competency_profile(competencies[comp_id])
            self.profiles.append(profile_text)
        
        # Tokenize profiles (simple whitespace tokenization)
        tokenized_profiles = [profile.lower().split() for profile in self.profiles]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_profiles)
    
    def retrieve(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-K competency candidates for query text.
        
        Args:
            query_text: Fragment text to search for
            top_k: Number of candidates to retrieve
            
        Returns:
            List of (competency_id, score) tuples sorted by score descending
        """
        # Tokenize query
        tokenized_query = query_text.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-K indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Return competency IDs and scores
        results = [(self.comp_ids[i], float(scores[i])) for i in top_indices]
        return results
    
    def retrieve_batch(self, texts: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Retrieve candidates for batch of texts.
        
        Args:
            texts: List of fragment texts
            top_k: Number of candidates per text
            
        Returns:
            List of candidate lists, one per input text
        """
        return [self.retrieve(text, top_k) for text in texts]

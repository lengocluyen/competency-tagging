"""LLM-based competency tagging with evidence extraction."""
import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import openai
from openai import OpenAI

import data_io


class LLMTagger:
    """OpenAI-based tagger for competencies with evidence extraction."""
    
    def __init__(
        self,
        competencies: Dict[str, Dict[str, Any]],
        model: str = "gpt-4o-mini",
        cache_dir: Optional[Path] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        validate_evidence: bool = True,
        repair_invalid: bool = True,
        api_key: Optional[str] = None,
    ):
        """Initialize LLM tagger.
        
        Args:
            competencies: Dictionary of competency_id -> competency record
            model: OpenAI model name
            cache_dir: Directory for caching LLM responses
            max_retries: Maximum number of retry attempts on rate limit
            retry_delay: Initial delay between retries (exponential backoff)
            validate_evidence: Whether to validate evidence quotes
            repair_invalid: Whether to attempt repair of invalid evidence
        """
        self.competencies = competencies
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.validate_evidence = validate_evidence
        self.repair_invalid = repair_invalid
        
        # Initialize OpenAI client
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set and no api_key provided to LLMTagger")
        self.client = OpenAI(api_key=api_key)
        
        # Setup cache
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = {}
            self._load_cache()
        else:
            self.cache = None
    
    def _load_cache(self) -> None:
        """Load cache from disk if exists."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "llm_cache.jsonl"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.cache[entry['key']] = entry['response']
    
    def _save_to_cache(self, key: str, response: Dict[str, Any]) -> None:
        """Save response to cache.
        
        Args:
            key: Cache key
            response: Response to cache
        """
        if not self.cache_dir:
            return
        
        self.cache[key] = response
        cache_file = self.cache_dir / "llm_cache.jsonl"
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'key': key, 'response': response}, ensure_ascii=False) + '\n')
    
    def _make_cache_key(self, fragment_text: str, candidates: List[Tuple[str, float]]) -> str:
        """Create cache key from inputs.
        
        Args:
            fragment_text: Fragment text
            candidates: List of (competency_id, score) tuples
            
        Returns:
            Cache key string
        """
        candidate_ids = [c[0] for c in candidates]
        return f"{hash(fragment_text)}_{hash(tuple(candidate_ids))}"
    
    def _build_prompt(self, fragment_text: str, candidates: List[Tuple[str, float]]) -> str:
        """Build prompt for LLM tagging.
        
        Args:
            fragment_text: Fragment text to tag
            candidates: List of (competency_id, score) tuples
            
        Returns:
            Prompt string
        """
        # Detect fragment language to choose appropriate competency labels/aliases
        lang = data_io.detect_language(fragment_text)

        prompt = f"""You are a competency tagging expert. Given a text fragment and a list of candidate competencies, identify which competencies are demonstrated in the fragment.

TEXT FRAGMENT:
{fragment_text}

CANDIDATE COMPETENCIES:
"""
        for comp_id, _ in candidates:
            comp = self.competencies[comp_id]
            label = data_io.get_competency_label_for_language(comp, lang)
            desc = comp.get('description', '')[:200]  # Truncate description (FR by default)
            prompt += f"- {comp_id}: {label}\n  {desc}\n"
        
        prompt += """
INSTRUCTIONS:
1. Identify which competencies are clearly demonstrated in the fragment
2. For each identified competency, extract the specific quote that provides evidence
3. Provide the exact character positions (start, end) of the quote in the fragment
4. Provide a confidence score (0.0-1.0) for each competency
5. If NONE of the candidates apply, set "none": true

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
        
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """Call OpenAI API with retry logic.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            Response text
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a competency tagging expert. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            
            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            
            except Exception as e:
                print(f"API call failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def _validate_evidence(self, fragment_text: str, evidence: Dict[str, Any]) -> bool:
        """Validate that evidence quote matches fragment text.
        
        Args:
            fragment_text: Original fragment text
            evidence: Evidence dictionary with quote, start_char, end_char
            
        Returns:
            True if valid, False otherwise
        """
        if not evidence or 'quote' not in evidence:
            return False
        
        quote = evidence['quote']
        start = evidence.get('start_char', 0)
        end = evidence.get('end_char', len(fragment_text))
        
        # Check bounds
        if start < 0 or end > len(fragment_text) or start >= end:
            return False
        
        # Check if quote matches
        extracted = fragment_text[start:end]
        return extracted == quote
    
    def _repair_evidence(self, fragment_text: str, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to repair invalid evidence by finding quote in text.
        
        Args:
            fragment_text: Original fragment text
            evidence: Invalid evidence dictionary
            
        Returns:
            Repaired evidence or None if repair failed
        """
        if 'quote' not in evidence:
            return None
        
        quote = evidence['quote']
        
        # Try to find quote in text
        idx = fragment_text.find(quote)
        if idx != -1:
            return {
                'quote': quote,
                'start_char': idx,
                'end_char': idx + len(quote)
            }
        
        return None
    
    def tag_fragment(
        self,
        fragment_text: str,
        candidates: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Tag a fragment with competencies.
        
        Args:
            fragment_text: Text to tag
            candidates: List of (competency_id, score) tuples from retrieval
            
        Returns:
            Tagging result with selected competencies, confidences, and evidence
        """
        # Check cache
        if self.cache is not None:
            cache_key = self._make_cache_key(fragment_text, candidates)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Build prompt and call API
        prompt = self._build_prompt(fragment_text, candidates)
        response_text = self._call_api(prompt)
        
        # Parse response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: return empty result
            result = {"selected": [], "none": True}
        
        # Validate and repair evidence
        if self.validate_evidence and 'selected' in result:
            # Normalize selected entries: allow either dicts or bare competency_id strings
            raw_selected = result.get('selected') or []
            if not isinstance(raw_selected, list):
                raw_selected = []

            normalized_selected = []
            for item in raw_selected:
                if isinstance(item, dict):
                    normalized_selected.append(item)
                elif isinstance(item, str):
                    normalized_selected.append({"competency_id": item})
                else:
                    continue

            validated_selected = []
            candidate_ids = {c[0] for c in candidates}

            for item in normalized_selected:
                comp_id = item.get('competency_id')

                # Check if competency_id is in candidates
                if comp_id not in candidate_ids:
                    continue

                evidence = item.get('evidence')

                # Validate evidence
                if evidence and not self._validate_evidence(fragment_text, evidence):
                    if self.repair_invalid:
                        # Attempt repair
                        repaired = self._repair_evidence(fragment_text, evidence)
                        if repaired:
                            item['evidence'] = repaired
                        else:
                            # Mark as invalid and skip
                            continue
                    else:
                        continue

                validated_selected.append(item)

            result['selected'] = validated_selected
        
        # Cache result
        if self.cache is not None:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def tag_batch(
        self,
        fragments: List[Tuple[str, List[Tuple[str, float]]]]
    ) -> List[Dict[str, Any]]:
        """Tag batch of fragments.
        
        Args:
            fragments: List of (fragment_text, candidates) tuples
            
        Returns:
            List of tagging results
        """
        results = []
        for fragment_text, candidates in fragments:
            result = self.tag_fragment(fragment_text, candidates)
            results.append(result)
        
        return results

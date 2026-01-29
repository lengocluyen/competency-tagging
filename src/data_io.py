"""Data I/O utilities for competency tagging pipeline."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries.
    
    Handles both standard JSONL (one JSON per line) and pretty-printed JSON arrays.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try to parse as multi-line JSON objects
    if content.startswith('['):
        # JSON array format
        return json.loads(content)
    else:
        # Try JSONL format or pretty-printed objects
        records = []
        current_obj = ""
        brace_count = 0
        
        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            
            current_obj += line + '\n'
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and current_obj.strip():
                try:
                    records.append(json.loads(current_obj))
                    current_obj = ""
                except json.JSONDecodeError:
                    pass
        
        return records


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save list of dictionaries to JSONL file.
    
    Args:
        data: List of dictionaries to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_competencies(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load competencies from JSONL file into dict keyed by competency_id.
    
    Args:
        path: Path to competencies_utc.jsonl
        
    Returns:
        Dictionary mapping competency_id to competency record
    """
    competencies = {}
    for record in load_jsonl(path):
        comp_id = record['competency_id']
        competencies[comp_id] = record
    return competencies


def load_fragments(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load fragments from JSONL file into dict keyed by fragment_id.
    
    Args:
        path: Path to <UV>_fragments.jsonl
        
    Returns:
        Dictionary mapping fragment_id to fragment record
    """
    fragments = {}
    for record in load_jsonl(path):
        frag_id = record['fragment_id']
        fragments[frag_id] = record
    return fragments


def load_gold_fragments(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load gold annotations from JSONL file.
    
    Args:
        path: Path to <UV>_gold_fragments.jsonl
        
    Returns:
        Dictionary mapping fragment_id to gold annotation record
    """
    gold = {}
    for record in load_jsonl(path):
        frag_id = record['fragment_id']
        gold[frag_id] = record
    return gold


def get_uv_from_path(path: Path) -> str:
    """Extract UV identifier from filename.
    
    Args:
        path: Path to file like AC02_fragments.jsonl
        
    Returns:
        UV identifier (e.g., 'AC02')
    """
    stem = path.stem

    # Many outputs are stored under per-UV directories with generic filenames
    # like: output_zero_shot_alluvs/AI14/fragment_predictions_zero_shot.jsonl
    # In these cases, the UV is the parent directory name.
    if stem.startswith("fragment_predictions"):
        parent = path.parent.name
        if parent:
            return parent

    return stem.split('_')[0]


def find_fragment_files(data_dir: Path) -> List[Path]:
    """Find all fragment files in data directory.
    
    Args:
        data_dir: Directory containing fragment files
        
    Returns:
        List of paths to fragment files
    """
    return sorted(data_dir.glob('*_fragments.jsonl'))


def find_gold_files(gold_dir: Path) -> List[Path]:
    """Find all gold annotation files in directory.
    
    Args:
        gold_dir: Directory containing gold fragment files
        
    Returns:
        List of paths to gold fragment files
    """
    return sorted(gold_dir.glob('*_gold_fragments.jsonl'))


def detect_language(text: str) -> str:
    """Very simple heuristic language detector for English vs French.

    This avoids adding external dependencies while being good enough
    for choosing competency labels/aliases in prompts.

    Args:
        text: Input text fragment

    Returns:
        "en" if the text looks mostly English, otherwise "fr".
    """
    lowered = text.lower()

    french_score = 0
    english_score = 0

    # Accented characters strongly indicate French
    if any(ch in lowered for ch in "àâçèéêëîïôùûüœ“”«»”’"):
        french_score += 3

    french_markers = [
        " le ", " la ", " les ", " des ", " une ", " un ",
        " et ", " ou ", " pour ", " avec ", " sans ", " sur ", " dans ",
    ]
    english_markers = [
        " the ", " and ", " or ", " of ", " to ", " with ",
        " without ", " in ", " on ", " for ",
    ]

    for token in french_markers:
        if token in lowered:
            french_score += 1

    for token in english_markers:
        if token in lowered:
            english_score += 1

    # Default to French if scores are tied but there are French accents
    if english_score > french_score:
        return "en"
    return "fr"


def get_competency_label_for_language(competency: Dict[str, Any], lang: str) -> str:
    """Return a human-readable label + aliases in the requested language.

    Args:
        competency: Competency record from competencies_utc.jsonl
        lang: "en" or "fr"

    Returns:
        String combining label and same-language aliases.
    """
    if lang == "en":
        base_label = competency.get("label_en") or competency.get("label", "")
        aliases = competency.get("aliases_en") or []
    else:
        base_label = competency.get("label_fr") or competency.get("label", "")
        aliases = competency.get("aliases_fr") or []

    alias_str = "; ".join(a for a in aliases if a)
    if alias_str:
        return f"{base_label} (aliases: {alias_str})"
    return base_label


def build_competency_profile(competency: Dict[str, Any]) -> str:
    """Build text profile for competency for retrieval.

    Uses both French and English labels/aliases so BM25 can match
    fragments written in either language.

    Args:
        competency: Competency record
        
    Returns:
        Combined text: labels (fr+en) + description + keywords + aliases (fr+en)
    """
    labels: List[str] = []
    # Prefer explicit language-specific labels when present
    if competency.get("label_fr"):
        labels.append(competency["label_fr"])
    if competency.get("label_en"):
        labels.append(competency["label_en"])
    # Fallback generic label
    if not labels and competency.get("label"):
        labels.append(competency["label"])

    parts = [" ".join(labels), competency.get("description", "")]

    # Add keywords
    if "keywords" in competency and competency["keywords"]:
        parts.append(" ".join(competency["keywords"]))

    # Add aliases in both languages if available
    aliases_fr = competency.get("aliases_fr") or []
    aliases_en = competency.get("aliases_en") or []
    all_aliases = []
    if aliases_fr:
        all_aliases.extend(aliases_fr)
    if aliases_en:
        all_aliases.extend(aliases_en)
    if all_aliases:
        parts.append(" ".join(all_aliases))

    return " ".join(p for p in parts if p).strip()

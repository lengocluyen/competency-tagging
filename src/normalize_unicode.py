import pathlib

BASE_DIR = pathlib.Path(__file__).parent

TARGET_DIRS = [
    BASE_DIR / "resources_fragments",
    BASE_DIR / "golds_fragments",
]

# Map of escaped sequences to their replacement characters
REPLACEMENTS = {
    "\\u3010": "[",
    "\\u3011": "]",
    "\\u2011": "-",  # non-breaking hyphen -> regular hyphen
    "\\u00a0": " ",  # non-breaking space -> regular space
    "\\u2018": "'",  # left single quotation mark -> apostrophe
    "\\u2019": "'",  # right single quotation mark -> apostrophe
    "\\u2013": "-",  # en dash -> regular hyphen
    "\\u2020": ":",  # dagger used in citations -> colon
    "\\u2014": "-",  # em dash -> regular hyphen
    "\\n": " ",  # keep JSONL one-line records; turn literal \\n into space
}


def normalize_file(path: pathlib.Path) -> bool:
    """Normalize unicode escape sequences in a single text file.

    Returns True if the file was modified, False otherwise.
    """
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip files that are not valid UTF-8
        return False

    modified = original
    for src, dst in REPLACEMENTS.items():
        modified = modified.replace(src, dst)

    if modified != original:
        path.write_text(modified, encoding="utf-8")
        return True
    return False


def main() -> None:
    total_files = 0
    changed_files = 0

    for d in TARGET_DIRS:
        if not d.exists():
            continue
        for path in d.rglob("*.jsonl"):
            total_files += 1
            if normalize_file(path):
                changed_files += 1
                print(f"Updated: {path.relative_to(BASE_DIR)}")

    print(f"Processed {total_files} JSONL files; modified {changed_files}.")


if __name__ == "__main__":
    main()

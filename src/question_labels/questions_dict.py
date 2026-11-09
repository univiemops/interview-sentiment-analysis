"""
Builds an *extended* question dictionary from the base dictionary by adding new, verified
example sentences taken from manually verified transcripts.

    data/questions_dict.json
        -> data/question_labels/questions_dict_extended.json

For every file in data/question_labels/verified/*_qid_verified.xlsx, each interviewer turn whose
verified label (column "verified_qid") is a known question ID and differs from the label
SBERT assigned ("qid_sbert") is treated as a *new* phrasing of that question. Its sentences
are appended to the corresponding list in the dictionary (deduplicated, ignoring very short
fragments). Re-running question_labels_sbert.py then benefits from the enriched dictionary,
which it picks up automatically if questions_dict_extended.json exists.

lodrantk 12/2025
"""
import re
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (scripts live in src/question_labels/, data/ is two levels up)
# ---------------------------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:  # e.g. when run in an interactive session
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
BASE_QUESTIONS_JSON = DATA_DIR / "questions_dict.json"
PROCESSED_DIR = DATA_DIR / "question_labels"
VERIFIED_DIR = PROCESSED_DIR / "verified"
EXTENDED_QUESTIONS_JSON = PROCESSED_DIR / "questions_dict_extended.json"

MIN_WORDS = 3  # ignore sentence fragments shorter than this


def normalize_questions_dict(dictionary):
    """Ensure every question ID maps to a list of example sentences."""
    return {qid: ([v] if isinstance(v, str) else list(v)) for qid, v in dictionary.items()}


def add_to_questions_dict(questions_dict):
    """Append verified, non-duplicate sentences from verified/ files to questions_dict."""
    if not VERIFIED_DIR.exists():
        print("No verified/ folder found - nothing to add.")
        return questions_dict

    verified_files = sorted(VERIFIED_DIR.glob("*_qid_verified.xlsx"))
    if not verified_files:
        print("No verified files found - nothing to add.")

    for file_path in verified_files:
        print("Adding data from:", file_path)
        df = pd.read_excel(file_path, engine="openpyxl")
        df.columns = [col.lower() for col in df.columns]  # normalize column names
        for idx, label in df["verified_qid"].items():
            if label in questions_dict:  # rows labeled with a valid question ID
                # A verified label that differs from the SBERT label means SBERT missed it,
                # so this turn contains new phrasings worth adding to the dictionary.
                if label != df.loc[idx, "qid_sbert"]:
                    question_text = df.loc[idx, "text"]
                    for sentence in re.split(r"(?<=[.!?]) +", str(question_text)):
                        if len(sentence.split()) >= MIN_WORDS and sentence not in questions_dict[label]:
                            questions_dict[label].append(sentence)
    return questions_dict


if __name__ == "__main__":
    # 1. Load the base dictionary (normalized to list-valued form)
    with open(BASE_QUESTIONS_JSON, "r", encoding="utf-8") as file:
        questions_dict = normalize_questions_dict(json.load(file))

    all_ids = list(questions_dict.keys())  # e.g. ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

    # 2. Add new verified sentences
    questions_dict = add_to_questions_dict(questions_dict)

    # 3. Save the extended dictionary
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXTENDED_QUESTIONS_JSON, "w", encoding="utf-8") as file:
        json.dump(questions_dict, file, indent=4, ensure_ascii=False)
    print("Wrote:", EXTENDED_QUESTIONS_JSON)

    # 4. Manually review the extended dictionary and remove any sentences that are
    #    unrelated to their question ID (an LLM could assist with this step).

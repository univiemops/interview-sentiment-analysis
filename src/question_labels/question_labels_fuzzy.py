"""
This script processes interview transcripts to identify and label the questions asked by
the interviewer using fuzzy string matching. Each interviewer turn is compared to a
predefined set of original questions (data/questions_dict.json) and assigned the question
ID of the most similar original question, provided the similarity score exceeds a
threshold. Matching uses the `fuzzywuzzy` library, which leverages the Levenshtein
distance. Results are written to new columns and saved to data/processed/fuzzy/.

This script was only used initially, to compare with the semantic SBERT matching
(question_labels_sbert.py). As SBERT clearly outperformed fuzzywuzzy and was not overly
computationally intensive, we focused on that method and this script is now obsolete. It
is kept here for reproducibility.

NOTE: The transcripts in data/transcripts_raw/ and data/questions_dict.json are simulated
example data. Each transcript is expected to contain a "Role" and a "Text" column, with
interviewer turns marked "Interviewer:".

lodrantk 12/2024
"""
import re
import json
from pathlib import Path

import pandas as pd
from fuzzywuzzy import process

# ---------------------------------------------------------------------------
# Paths (scripts live in src/question_labels/, data/ is two levels up)
# ---------------------------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:  # e.g. when run in an interactive session
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts_raw"
QUESTIONS_JSON = DATA_DIR / "questions_dict.json"
FUZZY_DIR = DATA_DIR / "question_labels" / "fuzzy"

INTERVIEWER_ROLE = "Interviewer:"
SCORE_THRESHOLD = 86


def get_subject_ids(transcripts_dir):
    """Return the sorted list of subject IDs (file stems) in the transcripts folder."""
    ids = [p.stem for p in transcripts_dir.glob("*.xlsx") if not p.name.startswith("~$")]
    return sorted(ids, key=lambda s: (re.sub(r"\d", "", s), int(re.sub(r"\D", "", s) or 0)))


def get_transcript_path(subject_id):
    return TRANSCRIPTS_DIR / f"{subject_id}.xlsx"


# Load the questions dictionary and invert it to {question_text: qid}.
# questions_dict.json maps {qid: question_text}; a value may also be a list of
# example phrasings, in which case every phrasing maps back to the same qid.
with open(QUESTIONS_JSON, "r", encoding="utf-8") as file:
    raw_questions = json.load(file)

question_to_id = {}
for qid, value in raw_questions.items():
    phrasings = [value] if isinstance(value, str) else list(value)
    for phrasing in phrasings:
        question_to_id[phrasing] = qid

original_questions = list(question_to_id.keys())


def get_best_match(question):
    """Return (qid, score) of the best-matching original question for a transcript turn."""
    individual_sentences = re.split(r"(?<=[.!?]) +", str(question))
    best_match_score = 0
    question_id = None
    for sentence in individual_sentences:
        match, score = process.extractOne(sentence, original_questions)
        if score > best_match_score:
            best_match_score = score
            question_id = question_to_id[match]
    return question_id, best_match_score


def label_questions(subject_id):
    file_path = get_transcript_path(subject_id)
    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [col.lower() for col in df.columns]  # normalize column names
    if "role" not in df.columns or "text" not in df.columns:
        raise KeyError(
            f"{file_path.name} must contain 'Role' and 'Text' columns; found {list(df.columns)}"
        )

    interviewer_mask = df["role"] == INTERVIEWER_ROLE

    matched_qids = []
    matching_scores = []
    for _, row in df.loc[interviewer_mask].iterrows():
        question_id, best_match_score = get_best_match(row["text"])
        if best_match_score > SCORE_THRESHOLD:
            matched_qids.append(question_id)
            matching_scores.append(best_match_score)
        else:
            matched_qids.append(None)
            matching_scores.append(None)

    df.loc[interviewer_mask, "qid_fuzzy"] = matched_qids
    df.loc[interviewer_mask, "match_score_fuzzy"] = matching_scores

    FUZZY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_excel(FUZZY_DIR / f"{subject_id}_qid_fuzzy.xlsx", index=False)
    print("Processed:", subject_id, "Found matches:", list(filter(None, matched_qids)))
    return df


if __name__ == "__main__":
    subjects = get_subject_ids(TRANSCRIPTS_DIR)
    print("Subjects found:", subjects)
    for subject_id in subjects:
        label_questions(subject_id)

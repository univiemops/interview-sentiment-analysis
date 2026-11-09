"""
This script processes interview transcripts to identify and label the key questions
asked by the interviewer. It uses a pre-trained Sentence-BERT (SBERT) model to compute
the semantic similarity between each interviewer turn in the transcript and a predefined
set of original questions (data/questions_dict.json). Each turn is assigned the question
ID of the most similar original question, provided the similarity score exceeds a
threshold. The question IDs and similarity scores are written to new columns and the
labeled transcript is saved to data/question_labels/sbert/.

NOTE: The transcripts in data/transcripts_raw/ (T0001.xlsx ... T0005.xlsx) and the
questions in data/questions_dict.json are simulated example data, provided so this code
runs end-to-end without the original (non-shareable) interview material. Each transcript
is expected to contain at least a "role" and a "text" column, where interviewer turns are
marked with the role "Interviewer:".

lodrantk 2/2025
"""
import re
import time
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths (scripts live in src/question_labels/, data/ is two levels up)
# ---------------------------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:  # e.g. when run in an interactive session
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts_raw"
PROCESSED_DIR = DATA_DIR / "question_labels"
SBERT_DIR = PROCESSED_DIR / "sbert"
TO_VERIFY_DIR = PROCESSED_DIR / "to_verify"

# Prefer the extended dictionary (built by questions_dict.py) if it exists,
# otherwise fall back to the base dictionary.
BASE_QUESTIONS_JSON = DATA_DIR / "questions_dict.json"
EXTENDED_QUESTIONS_JSON = PROCESSED_DIR / "questions_dict_extended.json"

INTERVIEWER_ROLE = "Interviewer:"
SCORE_THRESHOLD = 0.82

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # multilingual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_subject_ids(transcripts_dir):
    """Return the sorted list of subject IDs (file stems) in the transcripts folder."""
    ids = [p.stem for p in transcripts_dir.glob("*.xlsx") if not p.name.startswith("~$")]
    # e.g. T0001, T0002, ... -> sort by the trailing number
    return sorted(ids, key=lambda s: (re.sub(r"\d", "", s), int(re.sub(r"\D", "", s) or 0)))


def get_transcript_path(subject_id):
    return TRANSCRIPTS_DIR / f"{subject_id}.xlsx"


def normalize_questions_dict(dictionary):
    """Ensure every question ID maps to a list of example sentences."""
    return {qid: ([v] if isinstance(v, str) else list(v)) for qid, v in dictionary.items()}


def flip_dict(dictionary):
    """Flip {qid: [sentences]} into {sentence: qid} for faster lookup."""
    flipped = {}
    for qid, sentences in dictionary.items():
        for sentence in sentences:
            flipped[sentence] = qid
    return flipped


# ---------------------------------------------------------------------------
# Load questions dictionary and pre-compute embeddings
# ---------------------------------------------------------------------------
questions_path = EXTENDED_QUESTIONS_JSON if EXTENDED_QUESTIONS_JSON.exists() else BASE_QUESTIONS_JSON
with open(questions_path, "r", encoding="utf-8") as file:
    questions_dict = normalize_questions_dict(json.load(file))
print("Loaded questions from:", questions_path)

flipped_questions_dict = flip_dict(questions_dict)
original_questions = list(flipped_questions_dict.keys())
original_qids = list(flipped_questions_dict.values())
original_question_embeddings = model.encode(original_questions)


def label_questions(subject_id):
    start_time = time.time()
    file_path = get_transcript_path(subject_id)
    print("Processing:", file_path)

    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [col.lower() for col in df.columns]  # normalize column names
    if "role" not in df.columns or "text" not in df.columns:
        raise KeyError(
            f"{file_path.name} must contain 'Role' and 'Text' columns; found {list(df.columns)}"
        )

    interviewer_mask = df["role"] == INTERVIEWER_ROLE
    questions = df.loc[interviewer_mask, "text"].tolist()

    matched_qids = []
    matching_scores = []
    for interview_question in questions:
        sentences = re.split(r"(?<=[.!?]) +", str(interview_question))
        sentence_embeddings = model.encode(sentences)
        similarities = model.similarity(sentence_embeddings, original_question_embeddings)

        best_match_score = -1.0
        best_match_idx = 0
        for s in range(len(sentences)):
            j = int(similarities[s].argmax())
            score = float(similarities[s][j])
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j

        if best_match_score > SCORE_THRESHOLD:
            matched_qids.append(original_qids[best_match_idx])
            matching_scores.append(best_match_score)
        else:
            matched_qids.append(None)
            matching_scores.append(None)

    df.loc[interviewer_mask, "qid_sbert"] = matched_qids
    df.loc[interviewer_mask, "match_score_sbert"] = matching_scores

    # Label every question only once: keep each label where its score is maximal,
    # but keep duplicates when the score is very high (>= 0.95).
    max_scores = df.groupby("qid_sbert")["match_score_sbert"].transform("max")
    keep_mask = (df["match_score_sbert"] == max_scores) | (df["match_score_sbert"] >= 0.95)
    df["qid_sbert"] = df["qid_sbert"].where(keep_mask)
    df["match_score_sbert"] = df["match_score_sbert"].where(keep_mask)

    SBERT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SBERT_DIR / f"{subject_id}_qid_sbert.xlsx"
    df.to_excel(out_path, index=False)

    elapsed = time.time() - start_time
    print(f"Processed {subject_id}. Found matches: {df['qid_sbert'].dropna().tolist()}. "
          f"Elapsed time: {elapsed:.0f} s")
    return df


if __name__ == "__main__":
    subjects = get_subject_ids(TRANSCRIPTS_DIR)
    print("Subjects found:", subjects)

    # 1. Label questions and save to /sbert/
    for subject_id in subjects:
        label_questions(subject_id)

    # 2. Prepare a batch for verification: add empty columns for a human verifier to
    #    fill in, and save to /to_verify/
    TO_VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    for subject_id in subjects:
        sbert_path = SBERT_DIR / f"{subject_id}_qid_sbert.xlsx"
        if not sbert_path.exists():
            print("No labeled transcript for this subject:", subject_id)
            continue
        df = pd.read_excel(sbert_path, engine="openpyxl")
        df.insert(min(3, len(df.columns)), "verified_qid", None)
        df.insert(min(6, len(df.columns)), "verifier_comment", None)
        df.to_excel(TO_VERIFY_DIR / f"{subject_id}_qid_to_verify.xlsx", index=False)
        print("Saved to_verify:", subject_id)

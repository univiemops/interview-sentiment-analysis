#!/usr/bin/env python
"""
VADER/GerVADER/SentiWS Sentiment Analysis Script
=========================
lodrantk, 1/2026

Computes sentiment metrics for every condition in one run, writing two long-format files.

Conditions (every combination below is run):
  unit      "word" or "sentence"        (sentence mode only uses the English VADER analyzer)
  lexicon   "vader", "gervader", "sentiws"
  ambig_filter    True: filler words are removed before scoring; False: they are kept

Aggregation (computed for every condition and subject):
  mean    per-question metric, then averaged over the questions that have answers
  total   metric pooled over all of a subject's answers

Outputs (data/results/):
  sentiment_long.xlsx   one row per subject x condition x aggregation x metric
                        columns: subject_id, unit, lexicon, ambig_filter, aggregation, metric, value
  items/<subject>_items.xlsx   one file per subject, with one row per matched word
                        (word mode) or scored sentence (sentence mode), for all conditions
                        columns: unit, lexicon, ambig_filter, subject_id, qid, item, rating

To merge with clinician scores, see notebooks/merge_stipo_sentiment.ipynb.

Input: data/transcripts/<subject>_qid.xlsx
         columns: role ("Patient:" / ...), qid, text (German), text_english (English)
       data/lexicons/<lexicon file>   (not needed for sentence mode)
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------------------------------- #
# Paths and column names
# --------------------------------------------------------------------------- #
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
LEXICON_DIR = DATA_DIR / "lexicons"
RESULTS_DIR = DATA_DIR / "results"
ITEMS_DIR = RESULTS_DIR / "items"   # one items file per subject

QID_COLUMN = "qid"          # column holding question identifiers
ROLE_VALUE = "Patient:"     # role marking answer turns

# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
EN_FILLERS = ["yes", "no", "okay", "yeah", "like", "well"]
DE_FILLERS = ["einfach", "genau"]

AGGREGATIONS = ("mean", "total")

# Metrics produced for every subject, condition and aggregation.
METRICS = (
    "mean", "min", "max", "std", "diff_std",
    "positive_mean", "negative_mean", "pos_std", "neg_std",
    "n_units", "n_hits", "hit_ratio",
    "pos_ratio", "neg_ratio", "polarity_index", "valence_frequency",
)

# Reported only for the per-question "mean" aggregation. The pooled "total" has
# its own single min and max, which are not used.
PER_QUESTION_ONLY = ("min", "max")


@dataclass(frozen=True)
class Lexicon:
    text_col: str                  # transcript column the lexicon scores ("text" or "text_english")
    file: str | None               # file in LEXICON_DIR; None means the built-in VADER analyzer
    fillers: list[str] = field(default_factory=list)  # filler words removed when ambig_filter=True


LEXICONS: dict[str, Lexicon] = {
    "vader":    Lexicon("text_english", "vader_lexicon.txt",   EN_FILLERS),
    "gervader": Lexicon("text",         "GERVaderLexicon.txt", DE_FILLERS),
    "sentiws":  Lexicon("text",         "sentiws.csv",         DE_FILLERS),
}


@dataclass(frozen=True)
class Variant:
    unit: str        # "word" or "sentence"
    lexicon: str     # key into LEXICONS
    ambig_filter: bool     # True: remove filler words before scoring

    @property
    def name(self) -> str:
        return f"{self.unit}_{self.lexicon}_{'ambig_filter' if self.ambig_filter else 'no_ambig_filter'}"

    @property
    def is_sentence(self) -> bool:
        return self.unit == "sentence"

    @property
    def text_col(self) -> str:
        return LEXICONS[self.lexicon].text_col

    @property
    def fillers(self) -> list[str]:
        return LEXICONS[self.lexicon].fillers if self.ambig_filter else []


# Sentence scoring uses the English VADER analyzer only, so sentence mode is
# built for the "vader" lexicon alone. Every other combination is a word variant.
VARIANTS: list[Variant] = [
    Variant(unit, lexicon, ambig_filter)
    for unit in ("word", "sentence")
    for lexicon in LEXICONS
    for ambig_filter in (True, False)
    if not (unit == "sentence" and lexicon != "vader")
]


# --------------------------------------------------------------------------- #
# Lexicon / analyzer loading
# --------------------------------------------------------------------------- #
def load_scorer(variant: Variant) -> tuple[dict | None, SentimentIntensityAnalyzer | None]:
    """Return (lexicon_dict, analyzer). Exactly one is non-None.

    Sentence mode uses VADER's built-in SentimentIntensityAnalyzer.
    Word mode uses a {token: rating} dict loaded from the lexicon file.
    """
    lexicon = LEXICONS[variant.lexicon]
    if variant.is_sentence:
        return None, SentimentIntensityAnalyzer()

    path = LEXICON_DIR / lexicon.file
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file does not exist at {path}")

    if lexicon.file == "sentiws.csv":
        lex = pd.read_csv(path, skiprows=2, names=["Index", "V1", "V2"])
        lexicon_dict = dict(zip(lex["V1"], lex["V2"]))
    else:
        lex = pd.read_csv(
            path,
            delimiter="\t",
            header=None,
            names=["Token", "Rating", "Standard_deviation", "Raw_ratings"],
        )
        # case-sensitive keys on purpose (German nouns are capitalized)
        lexicon_dict = dict(zip(lex["Token"], lex["Rating"]))

    return lexicon_dict, None


# --------------------------------------------------------------------------- #
# Transcript loading
# --------------------------------------------------------------------------- #
def get_subject_ids() -> list[str]:
    """Return the sorted subject IDs found in the transcripts folder."""
    ids = []
    for f in TRANSCRIPTS_DIR.iterdir():
        match = re.match(r"(\w+)_qid\.xlsx", f.name)
        if match:
            ids.append(match.group(1))
    return sorted(ids)


def get_answers(subject_id: str) -> pd.DataFrame:
    """Load a subject's transcript and return only the patient answer turns.

    Warns when a patient turn has `text` but no `text_english`, and when a turn
    has `text` but no `qid`; then drops any turn missing `text`, `text_english`,
    or `qid` before analysis.
    """
    path = TRANSCRIPTS_DIR / f"{subject_id}_qid.xlsx"
    df = pd.read_excel(path, engine="openpyxl")

    answers = df[df["role"] == ROLE_VALUE].copy()

    # Warn: German text present but English translation missing.
    missing_en = answers["text"].notna() & answers["text_english"].isna()
    if missing_en.any():
        qids = answers.loc[missing_en, QID_COLUMN].tolist()
        print(f"\n  WARNING: {subject_id}: {int(missing_en.sum())} patient row(s) "
              f"have 'text' but no 'text_english' (qids: {qids})")

    # Warn: text present but qid missing.
    missing_qid = answers["text"].notna() & answers[QID_COLUMN].isna()
    if missing_qid.any():
        print(f"\n  WARNING: {subject_id}: {int(missing_qid.sum())} patient row(s) "
              f"have 'text' but no '{QID_COLUMN}'")

    # Drop rows missing any of text, text_english, or qid before any analysis.
    return answers.dropna(subset=["text", "text_english", QID_COLUMN]).copy()


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def get_sentences(text: str, fillers: list[str]) -> list[str]:
    """Split text into sentences, stripping filler words."""
    tokens = re.split(r"(?<=[.!?]) +", str(text).strip())
    tokens = [t for t in tokens if t]

    cleaned = []
    for t in tokens:
        for aw in fillers:
            if re.search(rf"\b{aw}\b", t, re.IGNORECASE):
                t = re.sub(rf"\b{aw}\b", "", t, flags=re.IGNORECASE).strip()
        if t:
            cleaned.append(t)
    return cleaned


def get_words(text: str, fillers: list[str]) -> tuple[list[str], list[bool]]:
    """Return (words, is_first_in_sentence flags) with filler words removed.

    The flag lets the scorer try a capitalized sentence-initial word both as-is
    and lowercased, while preserving case for mid-sentence German nouns.
    """
    words: list[str] = []
    tags: list[bool] = []
    for sent in get_sentences(text, fillers):
        toks = re.findall(r"[A-Za-zÄÖÜäöüß]+(?:'[A-Za-zÄÖÜäöüß]+)?", str(sent))
        for i, tok in enumerate(toks):
            words.append(tok)
            tags.append(i == 0)
    return words, tags


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_words(words: list[str], tags: list[bool], lexicon_dict: dict) -> list[tuple[str, float]]:
    """Look up each word in the lexicon. Return (matched_word, rating) per hit."""
    hits = []
    for w, is_first in zip(words, tags):
        # sentence-initial: try the capitalized form, then lowercase
        key = w if (w in lexicon_dict) else (w.lower() if is_first else w)
        if key in lexicon_dict:
            hits.append((key, float(lexicon_dict[key])))
    return hits


def score_items(text: str, variant: Variant, lexicon_dict, analyzer) -> tuple[int, list[tuple[str, float]]]:
    """Score one block of text. Return (n_units, hits).

    n_units counts words (word mode) or sentences (sentence mode) after filler
    removal. hits holds one (item, rating) pair per scored item: the matched word,
    or the sentence with a non-zero compound score. Neutral sentences are dropped,
    as words absent from the lexicon are.
    """
    if variant.is_sentence:
        sentences = get_sentences(text, variant.fillers)
        scored = [(s, analyzer.polarity_scores(s)["compound"]) for s in sentences]
        return len(sentences), [(s, r) for s, r in scored if r != 0]

    words, tags = get_words(text, variant.fillers)
    return len(words), score_words(words, tags, lexicon_dict)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarise(n_units: int, hits: list[tuple[str, float]]) -> dict[str, float]:
    """Compute every metric for one block of text (one question, or all answers)."""
    ratings = np.array([r for _, r in hits], dtype=float)
    pos = ratings[ratings > 0]
    neg = ratings[ratings < 0]

    def stat(fn, values, min_size):
        """Apply fn when there are at least min_size values, otherwise NaN."""
        return float(fn(values)) if values.size >= min_size else np.nan

    def ratio(count):
        return count / n_units if n_units else np.nan

    return {
        "mean":           stat(np.mean, ratings, 1),
        "min":            stat(np.min, ratings, 1),
        "max":            stat(np.max, ratings, 1),
        "std":            stat(lambda x: np.std(x, ddof=1), ratings, 2),
        "diff_std":       stat(lambda x: np.std(np.diff(x), ddof=1), ratings, 3),
        "positive_mean":  stat(np.mean, pos, 1),
        "negative_mean":  stat(np.mean, neg, 1),
        "pos_std":        stat(lambda x: np.std(x, ddof=1), pos, 2),
        "neg_std":        stat(lambda x: np.std(x, ddof=1), neg, 2),
        "n_units":        n_units,
        "n_hits":         ratings.size,
        "hit_ratio":      ratio(ratings.size),
        "pos_ratio":      ratio(pos.size),
        "neg_ratio":      ratio(neg.size),
        "polarity_index": ratio(pos.size) - ratio(neg.size),
        "valence_frequency": ratio(pos.size) + ratio(neg.size),
    }


def mean_ignoring_nan(values) -> float:
    """Average the values, skipping NaN. Returns NaN if nothing is left."""
    kept = [v for v in values if not np.isnan(v)]
    return float(np.mean(kept)) if kept else np.nan


def aggregate_mean(question_summaries: list[dict]) -> dict[str, float]:
    """Per-question summaries averaged over the questions that have answers.

    Questions with no answer units are left out, so the average covers only the
    questions that were answered.
    """
    answered = [q for q in question_summaries if q["n_units"] > 0]
    return {m: mean_ignoring_nan([q[m] for q in answered]) for m in METRICS}


# --------------------------------------------------------------------------- #
# Per-subject processing
# --------------------------------------------------------------------------- #
def process_subject(subject_id: str, variant: Variant, lexicon_dict, analyzer):
    """Score one subject under one variant.

    Returns (summary_rows, item_rows). Both are lists of dicts ready for a DataFrame.
    """
    df = get_answers(subject_id)
    condition = {"unit": variant.unit, "lexicon": variant.lexicon, "ambig_filter": variant.ambig_filter}

    # Per question: score the question's answers together.
    question_summaries = []
    item_rows = []
    for qid in df[QID_COLUMN].unique():
        q_text = df.loc[df[QID_COLUMN] == qid, variant.text_col].str.cat(sep=" ")
        n_units, hits = score_items(q_text, variant, lexicon_dict, analyzer)
        question_summaries.append(summarise(n_units, hits))
        item_rows += [
            {**condition, "subject_id": subject_id, "qid": qid, "item": item, "rating": rating}
            for item, rating in hits
        ]

    # Pooled: score all of the subject's answers as one text.
    pooled_text = df[variant.text_col].str.cat(sep=" ")
    pooled = summarise(*score_items(pooled_text, variant, lexicon_dict, analyzer))

    results = {"mean": aggregate_mean(question_summaries), "total": pooled}
    summary_rows = [
        {**condition, "subject_id": subject_id, "aggregation": agg, "metric": metric, "value": values[metric]}
        for agg in AGGREGATIONS
        for values in [results[agg]]
        for metric in METRICS
        if agg == "mean" or metric not in PER_QUESTION_ONLY
    ]
    return summary_rows, item_rows


# --------------------------------------------------------------------------- #
# Run all conditions
# --------------------------------------------------------------------------- #
def main() -> None:
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    subjects = get_subject_ids()
    print(f"Subjects: {len(subjects)} · conditions: {len(VARIANTS)}")

    summary_rows: list[dict] = []
    scorers: dict[tuple[str, str], tuple] = {}  # cache: load each lexicon once

    # Subject-major loop: all conditions for one subject are scored together, so
    # that subject's items file can be written as soon as it is done.
    for i, subject_id in enumerate(subjects, 1):
        print(f"\n##### [{i}/{len(subjects)}] {subject_id}")
        item_rows: list[dict] = []
        for variant in VARIANTS:
            key = (variant.unit, variant.lexicon)
            if key not in scorers:
                scorers[key] = load_scorer(variant)
            lexicon_dict, analyzer = scorers[key]

            print(f"=== {variant.name} ===", end=" ")
            try:
                rows, items = process_subject(subject_id, variant, lexicon_dict, analyzer)
                summary_rows += rows
                item_rows += items
                print("ok")
            except Exception as e:  # keep going; report which subject failed
                print(f"Error: {e}")
                traceback.print_exc()

        items_path = ITEMS_DIR / f"{subject_id}_items.xlsx"
        pd.DataFrame(item_rows).to_excel(items_path, index=False, engine="openpyxl")
        print(f"Wrote {items_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "sentiment_long.xlsx"
    pd.DataFrame(summary_rows).to_excel(summary_path, index=False, engine="openpyxl")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()

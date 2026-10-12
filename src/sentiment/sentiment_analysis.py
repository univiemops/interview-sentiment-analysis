#!/usr/bin/env python
"""
VADER/GerVADER/SentiWS Sentiment Analysis Script
=========================
lodrantk, 1/2026

Computes word-level or sentence-level sentiment metrics from per-subject
transcript spreadsheets, two ways:

  *_mean   -> per-question metric, then averaged over questions
  *_total  -> pooled over all of a subject's answers

One output file per variant is written to:  data/results/sentiment_<test_id>.xlsx
with one row per subject.

Variants (language x unit x edited/unedited) are declared once in VARIANTS below.
"Edited" means the variant's `ambiguous` filler words are stripped from the
text before scoring; "unedited" passes an empty filler list. Run one variant with
--test-id, or all of them by default.

To merge with clinician scores, see notebooks/merge_stipo_sentiment.ipynb.

Input  : data/transcripts/<subject>_qid.xlsx
           columns: role ("Patient:" / ...), qid, text (German), english_text (English)
         data/lexicons/<lexicon file>   (not needed for the "sentence" variants)
Output : data/results/sentiment_<test_id>.xlsx   (one row per subject)
"""

from __future__ import annotations

import argparse
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
LEXICON_DIR = DATA_DIR / "lexicons"
RESULTS_DIR = DATA_DIR / "results"

QID_COLUMN = "qid"          # column holding question identifiers
ROLE_VALUE = "Patient:"     # role marking answer turns

# --------------------------------------------------------------------------- #
# Variant configuration -- the single source of truth for "which analysis"
# --------------------------------------------------------------------------- #
EN_FILLERS = ["yes", "no", "okay", "yeah", "like", "well"]
DE_FILLERS = ["einfach", "genau"]


@dataclass(frozen=True)
class Variant:
    test_id: str
    unit: str                       # "word" or "sentence"
    text_col: str                   # column to score ("text" or "english_text")
    lexicon_file: str | None        # filename in LEXICON_DIR, or None -> built-in VADER analyzer
    ambiguous: list[str] = field(default_factory=list)

    @property
    def is_sentence(self) -> bool:
        return self.unit == "sentence"


VARIANTS: dict[str, Variant] = {
    "vader":             Variant("vader",             "word",     "english_text", "vader_lexicon.txt",    EN_FILLERS),
    "vader_unedited":    Variant("vader_unedited",    "word",     "english_text", "vader_lexicon.txt",    []),
    "gervader":          Variant("gervader",          "word",     "text",         "GERVaderLexicon.txt",  DE_FILLERS),
    "gervader_unedited": Variant("gervader_unedited", "word",     "text",         "GERVaderLexicon.txt",  []),
    "sentiws":           Variant("sentiws",           "word",     "text",         "sentiws.csv",          DE_FILLERS),
    "sentence":          Variant("sentence",          "sentence", "english_text", None,                   EN_FILLERS),
    "sentence_unedited": Variant("sentence_unedited", "sentence", "english_text", None,                   []),
}


# --------------------------------------------------------------------------- #
# Lexicon / analyzer loading
# --------------------------------------------------------------------------- #
def load_scorer(variant: Variant) -> tuple[dict | None, SentimentIntensityAnalyzer | None]:
    """Return (lexicon_dict, analyzer). Exactly one is non-None.

    Sentence variants use VADER's built-in SentimentIntensityAnalyzer.
    Word variants use a {token: rating} dict loaded from the lexicon file.
    """
    if variant.lexicon_file is None:
        return None, SentimentIntensityAnalyzer()

    path = LEXICON_DIR / variant.lexicon_file
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file does not exist at {path}")

    if variant.lexicon_file == "sentiws.csv":
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
def get_subject_id(filename: str) -> str | None:
    match = re.match(r"(\w+)_qid\.xlsx", filename)
    if match:
        return match.group(1)
    print(f"No subject ID found in filename: {filename}")
    return None

def get_answers(subject_id: str, variant: Variant) -> pd.DataFrame:
    """Load a subject's transcript and return only the patient answer turns.

    Warns when a patient turn has `text` but no `english_text`, and when a turn
    has `text` but no `qid`; then drops any turn missing `text`, `english_text`,
    or `qid` before analysis.
    """
    path = TRANSCRIPTS_DIR / f"{subject_id}_qid.xlsx"
    df = pd.read_excel(path, engine="openpyxl")

    answers = df[df["role"] == ROLE_VALUE].copy()

    # Warn: German text present but English translation missing.
    missing_en = answers["text"].notna() & answers["english_text"].isna()
    if missing_en.any():
        qids = answers.loc[missing_en, QID_COLUMN].tolist()
        print(f"\n  WARNING: {subject_id}: {int(missing_en.sum())} patient row(s) "
              f"have 'text' but no 'english_text' (qids: {qids})")

    # Warn: text present but qid missing.
    missing_qid = answers["text"].notna() & answers[QID_COLUMN].isna()
    if missing_qid.any():
        print(f"\n  WARNING: {subject_id}: {int(missing_qid.sum())} patient row(s) "
              f"have 'text' but no '{QID_COLUMN}'")

    # Drop rows missing any of text, english_text, or qid before any analysis.
    return answers.dropna(subset=["text", "english_text", QID_COLUMN]).copy()

# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def get_sentences(text: str, ambiguous: list[str]) -> list[str]:
    """Split text into sentences, stripping ambiguous filler words."""
    tokens = re.split(r"(?<=[.!?]) +", str(text).strip())
    tokens = [t for t in tokens if t]

    cleaned = []
    for t in tokens:
        for aw in ambiguous:
            if re.search(rf"\b{aw}\b", t, re.IGNORECASE):
                t = re.sub(rf"\b{aw}\b", "", t, flags=re.IGNORECASE).strip()
        if t:
            cleaned.append(t)
    return cleaned


def get_words(text: str, ambiguous: list[str]) -> tuple[list[str], list[bool]]:
    """Return (words, is_first_in_sentence flags) with ambiguous words removed.

    The flag lets the scorer try a capitalized sentence-initial word both as-is
    and lowercased, while preserving case for mid-sentence German nouns.
    """
    words: list[str] = []
    tags: list[bool] = []
    for sent in get_sentences(text, ambiguous):
        toks = re.findall(r"[A-Za-zÄÖÜäöüß]+(?:'[A-Za-zÄÖÜäöüß]+)?", str(sent))
        for i, tok in enumerate(toks):
            words.append(tok)
            tags.append(i == 0)
    return words, tags


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_words(words: list[str], tags: list[bool], lexicon_dict: dict) -> tuple[list[float], list[str]]:
    """Look up each word in the lexicon; return (ratings, matched_words)."""
    hits, matched = [], []
    for w, is_first in zip(words, tags):
        # sentence-initial: try the capitalized form, then lowercase
        key = w if (w in lexicon_dict) else (w.lower() if is_first else w)
        if key in lexicon_dict:
            hits.append(float(lexicon_dict[key]))
            matched.append(key)
    return hits, matched


def score_sentences(sentences: list[str], analyzer: SentimentIntensityAnalyzer) -> list[float]:
    """Compound score per sentence, dropping neutral (0.0) sentences.

    Dropping zeros parallels the word-wise analysis, where words absent from the
    lexicon simply produce no rating.
    """
    scores = [analyzer.polarity_scores(s)["compound"] for s in sentences]
    return [s for s in scores if s != 0]


def units_and_ratings(text: str, variant: Variant, lexicon_dict, analyzer):
    """Return (unit_count, ratings_array, matched_words) for a block of text.

    'unit' is sentences in sentence mode, words in word mode. `matched_words`
    is empty in sentence mode.
    """
    if variant.is_sentence:
        sentences = get_sentences(text, variant.ambiguous)
        hits = score_sentences(sentences, analyzer)
        return len(sentences), np.asarray(hits, dtype=float), []

    words, tags = get_words(text, variant.ambiguous)
    hits, matched = score_words(words, tags, lexicon_dict)
    return len(words), np.asarray(hits, dtype=float), matched


# --------------------------------------------------------------------------- #
# Per-subject processing
# --------------------------------------------------------------------------- #
def process_subject(subject_id: str, variant: Variant, lexicon_dict, analyzer) -> dict:
    out: dict = {"subject_id": subject_id}
    df = get_answers(subject_id, variant)

    # NOTE: the *_word_* column names below also count sentences in sentence mode.
    # Kept as-is to preserve the output schema the merge notebook expects.

    # ---- collect per-question arrays --------------------------------------
    per_q_unit_counts: list[int] = []   # sentences or words per Q
    per_q_ratings: list[np.ndarray] = []  # rating array per Q (may be empty)

    for qid in df[QID_COLUMN].unique():
        q_text = df[df[QID_COLUMN] == qid][variant.text_col].str.cat(sep=" ")
        n_units, ratings, _ = units_and_ratings(q_text, variant, lexicon_dict, analyzer)
        per_q_unit_counts.append(n_units)
        per_q_ratings.append(ratings)

    per_q_unit_counts = np.asarray(per_q_unit_counts)
    nonempty = [r for r in per_q_ratings if r.size > 0]

    # ---- per-Q averaged metrics (only Qs that produced ratings) -----------
    std_terms      = [np.std(r, ddof=1)          for r in per_q_ratings if r.size > 1]
    diff_std_terms = [np.std(np.diff(r), ddof=1) for r in per_q_ratings if r.size > 2]
    neg_mean_terms = [np.mean(r[r < 0])          for r in per_q_ratings if np.any(r < 0)]
    pos_mean_terms = [np.mean(r[r > 0])          for r in per_q_ratings if np.any(r > 0)]
    # guard on the COUNT of neg/pos values (was mistakenly (r < 0).size before),
    # and use ddof=1 to match the other std terms
    neg_std_terms  = [np.std(r[r < 0], ddof=1)   for r in per_q_ratings if (r < 0).sum() > 1]
    pos_std_terms  = [np.std(r[r > 0], ddof=1)   for r in per_q_ratings if (r > 0).sum() > 1]

    out["std_mean"]           = float(np.mean(std_terms))      if std_terms      else np.nan
    out["neg_std_mean"]       = float(np.mean(neg_std_terms))  if neg_std_terms  else np.nan
    out["pos_std_mean"]       = float(np.mean(pos_std_terms))  if pos_std_terms  else np.nan
    out["diff_std_mean"]      = float(np.mean(diff_std_terms)) if diff_std_terms else np.nan
    out["negative_mean_mean"] = float(np.mean(neg_mean_terms)) if neg_mean_terms else np.nan
    out["positive_mean_mean"] = float(np.mean(pos_mean_terms)) if pos_mean_terms else np.nan

    out["mean_mean"] = float(np.mean([np.mean(r) for r in nonempty])) if nonempty else np.nan
    out["min_mean"]  = float(np.mean([np.min(r)  for r in nonempty])) if nonempty else np.nan
    out["max_mean"]  = float(np.mean([np.max(r)  for r in nonempty])) if nonempty else np.nan

    # ---- ratios: well-defined even with 0 hits -> include all answered Qs --
    answered = per_q_unit_counts > 0
    unit_counts = per_q_unit_counts[answered]
    lex_counts = np.array([r.size for r in per_q_ratings])[answered]
    neg_counts = np.array([(r < 0).sum() for r in per_q_ratings])[answered]
    pos_counts = np.array([(r > 0).sum() for r in per_q_ratings])[answered]

    neg_ratio_per_q = neg_counts / unit_counts
    pos_ratio_per_q = pos_counts / unit_counts
    lex_ratio_per_q = lex_counts / unit_counts

    out["neg_ratio_mean"]          = float(np.mean(neg_ratio_per_q)) if neg_ratio_per_q.size else np.nan
    out["pos_ratio_mean"]          = float(np.mean(pos_ratio_per_q)) if pos_ratio_per_q.size else np.nan
    out["polarity_index_mean"]     = float(np.mean(pos_ratio_per_q - neg_ratio_per_q)) if pos_ratio_per_q.size else np.nan
    out["lexicon_word_ratio_mean"] = float(np.mean(lex_ratio_per_q)) if lex_ratio_per_q.size else np.nan
    out["valence_frequency_mean"]  = float(np.mean(pos_ratio_per_q + neg_ratio_per_q)) if pos_ratio_per_q.size else np.nan

    out["lexicon_word_count_mean"]  = float(np.mean(lex_counts))  if lex_counts.size  else np.nan
    out["response_word_count_mean"] = float(np.mean(unit_counts)) if unit_counts.size else np.nan

    # ---- pooled total across all answers ----------------------------------
    full_text = df[variant.text_col].str.cat(sep=" ")
    unit_count, ratings, matched = units_and_ratings(full_text, variant, lexicon_dict, analyzer)
    lex_count = ratings.size
    pos_vals = ratings[ratings > 0]
    neg_vals = ratings[ratings < 0]

    out["lexicon_ratings"] = ratings.tolist()   # stored as a string repr in Excel;
    out["lexicon_words"]   = matched            # the merge notebook literal_eval()s these back
    out["response_word_count_total"] = unit_count
    out["lexicon_word_count_total"]  = int(lex_count)

    out["mean_total"]     = float(np.mean(ratings))                 if lex_count      else np.nan
    out["min_total"]      = float(np.min(ratings))                  if lex_count      else np.nan
    out["max_total"]      = float(np.max(ratings))                  if lex_count      else np.nan
    out["std_total"]      = float(np.std(ratings, ddof=1))          if lex_count > 1  else np.nan
    out["diff_std_total"] = float(np.std(np.diff(ratings), ddof=1)) if lex_count > 2  else np.nan

    out["positive_mean_total"] = float(np.mean(pos_vals)) if pos_vals.size else np.nan
    out["negative_mean_total"] = float(np.mean(neg_vals)) if neg_vals.size else np.nan
    # guard on the count of the values actually being std'd (was lex_count > 1)
    out["neg_std_total"] = float(np.std(neg_vals, ddof=1)) if neg_vals.size > 1 else np.nan
    out["pos_std_total"] = float(np.std(pos_vals, ddof=1)) if pos_vals.size > 1 else np.nan

    out["lexicon_word_ratio_total"] = float(lex_count / unit_count) if unit_count else 0.0
    out["neg_ratio_total"]          = float(neg_vals.size / unit_count) if unit_count else 0.0
    out["pos_ratio_total"]          = float(pos_vals.size / unit_count) if unit_count else 0.0
    out["polarity_index_total"]     = float((pos_vals.size - neg_vals.size) / unit_count) if unit_count else 0.0
    out["valence_frequency_total"]  = float((pos_vals.size + neg_vals.size) / unit_count) if unit_count else 0.0

    return out


# --------------------------------------------------------------------------- #
# Variant runner
# --------------------------------------------------------------------------- #
def run_variant(variant: Variant) -> Path:
    print(f"\n=== Variant: {variant.test_id} "
          f"(unit={variant.unit}, text_col={variant.text_col}, "
          f"lexicon={variant.lexicon_file or 'built-in VADER'}) ===")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lexicon_dict, analyzer = load_scorer(variant)

    subjects = sorted(
        s for f in TRANSCRIPTS_DIR.iterdir()
        if f.name.endswith("_qid.xlsx") and (s := get_subject_id(f.name))
    )
    print(f"Total subjects: {len(subjects)}")

    results = []
    for i, subject_id in enumerate(subjects):
        print(f"[{i + 1}/{len(subjects)}] {subject_id}", end=" ")
        try:
            results.append(process_subject(subject_id, variant, lexicon_dict, analyzer))
            print("Yaay")
        except Exception as e:  # keep going; report which subject failed
            print(f"Error: {e}")
            traceback.print_exc()

    out_path = RESULTS_DIR / f"sentiment_{variant.test_id}.xlsx"
    pd.DataFrame(results).to_excel(out_path, index=False, engine="openpyxl")
    print(f"Wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sentiment metrics per subject.")
    parser.add_argument(
        "--test-id",
        choices=[*VARIANTS.keys(), "all"],
        default="all",
        help="Which lexicon/approach variant to run (default: all).",
    )
    args = parser.parse_args()

    to_run = list(VARIANTS.values()) if args.test_id == "all" else [VARIANTS[args.test_id]]
    for variant in to_run:
        run_variant(variant)


if __name__ == "__main__":
    main()

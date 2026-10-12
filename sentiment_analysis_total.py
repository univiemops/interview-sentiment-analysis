# ./splitting_env/bin/python src/sentiment_analysis_total.py
#
import pandas as pd
import re
import os
import numpy as np
from pathlib import Path
from ast import literal_eval
import pandas as pd
from itertools import chain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
"""
K.Lodrant 10/2025

Sentiment Analysis Script
Unedited: keeps original lexions; otherwise removes ambiguous words from text prior to analysis.
Saves sentiment metrics to Excel file: results/sentiment_<test_id>.xlsx with one row per subject.
To merge with clinician scores and for different test_ids, use notebooks/merge_stipo_sentiment.ipynb.
"""
test_id = "gervader"  # choose the lexicon to use for sentiment analysis
# options: "vader", "gervader", "sentiws", "unedited", "sentence", "sentence_unedited"

# Get the current working directory, root directory, and data directory
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
qid_column = "qid"  # name of column storing question identifiers ('qids')
RESULTS_DIR = DATA_DIR / "results"
print(f"Using {test_id} lexicon.")

# Flags
IS_SENTENCE = test_id in {"sentence", "sentence_unedited"}
IS_ENGLISH  = test_id in {"vader", "vader_unedited", "sentence", "sentence_unedited"}
IS_GERMAN   = test_id in {"gervader", "sentiws", "gervader_unedited"}
IS_UNEDITED = test_id in {"gervader_unedited", "vader_unedited", "sentence_unedited"}

# Language-specific setup
if IS_GERMAN:
    TEXT_DIR = DATA_DIR / "transcripts_raw" / "english"
    text_col = "text"
    lexicon_path = (
        DATA_DIR / "lexicons" / "sentiws.csv"
        if test_id == "sentiws"
        else DATA_DIR / "lexicons" / "GERVaderLexicon.txt"
    )
    ambiguous_words = ["einfach", "genau"]

elif IS_ENGLISH:
    TEXT_DIR = DATA_DIR / "transcripts_raw" / "english" #english_edited
    text_col = "english_text"
    lexicon_path = DATA_DIR / "lexicons" / "vader_lexicon.txt"
    ambiguous_words = ["yes", "no", "okay", "yeah", "like", "well"]#, "want","kind", 

# Adjustments
if IS_SENTENCE:
    ANALYZER = SentimentIntensityAnalyzer()

if IS_UNEDITED:
    # use unedited text column
    #text_col = "english_text"
    ambiguous_words = []  # do not remove any words

# Create Folders for Processed Data
directories = [
    RESULTS_DIR
]
for directory in directories:
    try:
        os.makedirs(directory, exist_ok=True)
        # print(f"Created (or already exists): {directory}")
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")

# Load Lexicon to lexicon_dict
if not lexicon_path.exists():
    print(f"Error: Lexicon file does not exist at {lexicon_path}.")
else:
    try:
        if test_id == "sentiws":
            lexicon = pd.read_csv(lexicon_path, skiprows=2, names=["Index", "V1", "V2"])
            lexicon_dict = dict(zip(lexicon["V1"], lexicon["V2"]))
        else:
            lexicon = pd.read_csv(
                lexicon_path,
                delimiter="\t",
                header=None,
                names=["Token", "Rating", "Standard_deviation", "Raw_ratings"],
            )
            lexicon_dict = dict(  # create dict {Token: Rating}
                zip(lexicon["Token"], lexicon["Rating"])  #not .lower(), must be case sensitive
            )
    except Exception as e:
        print(f"Error loading lexicon file: {e}")

# Get subject ID from verified file
def get_subject_id(path):
    pattern = r"(\w+)_qid.xlsx"
    match = re.match(pattern, path)
    if match:
        return match.group(1)
    else:
        print(f"No subject ID found in the path: {path}")
        return None

def get_transcript_df(subject_id):
    """Get DataFrame of the verified transcript (with all questions properly labeled) for subject."""
    file_path = TEXT_DIR / f"{subject_id}_qid.xlsx"
    print(file_path)
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        return df
    except Exception as e:
        print(f"Error loading verified file: {e}")
        return None

def get_all_answers(df, qid_column=qid_column):
    """Get all answers (from main question and sub-questions) with labeled ids."""
    answers_df = df[df["role"] == "Patient:"]
    #answers_df = answers_df[
    #    answers_df[qid_column].notna()
    #]  # Keep only labeled == identity portion of the interview
    return answers_df

def get_sentences(text):
    """Split text into sentences using regex."""
    tokens = re.split(r'(?<=[.!?]) +', str(text).strip())
    tokens = [str(t) for t in tokens if t]  # Remove empty strings

    filtered_tokens = []
    for t in tokens: #remove ambiguous words from sentences
        for aw in ambiguous_words:
            if re.search(rf'\b{aw}\b', t, re.IGNORECASE):
                t = re.sub(rf'\b{aw}\b', '', t, flags=re.IGNORECASE).strip()
        if t:  # Only add non-empty sentences
            filtered_tokens.append(t)

    return filtered_tokens

def extract_sentence_ratings(tokens, score):
    all_scores = [ANALYZER.polarity_scores(s)[score] for s in tokens]
    return [s for s in all_scores if s != 0] # remove neutral sentences, parallel with word-wise analysis

def extract_word_ratings(tokens, tags):
    hits = []
    words = []
    for i, w in enumerate(tokens):
        if tags[i]: # first word in sentence
            key = w if w in lexicon_dict else w.lower() # try capitalized else lowercase
        else:
            key = w # if not first word, keep case (for German nouns)
        if key in lexicon_dict:
            hits.append(float(lexicon_dict[key]))
            words.append(key) # save the case that matches the lexicon
    return hits, words

def get_tokens(text):
    if IS_SENTENCE:
        return get_sentences(text)
    else:
        sentences = get_sentences(text) # This filters out ambiguous words at sentence level
        words = []
        tags = []
        for sent in sentences:
            tokens = re.findall(r"[A-Za-zÄÖÜäöüß]+(?:'[A-Za-zÄÖÜäöüß]+)?", str(sent)) #“word” ≈ “alphabetic word”
            for i, tok in enumerate(tokens):
                words.append(tok)
                tags.append(i == 0)  # True if first word in sentence, else False
        return words, tags
    
def process_subject(subject_id):
    subject_sentiment = {"subject_id": subject_id}
    subject_df = get_transcript_df(subject_id)
    df = get_all_answers(subject_df)

    qids = df[qid_column].unique()

    per_q_unit_counts = []   # sentences if IS_SENTENCE else words
    per_q_lex_counts  = []   # count of non-neutral sentence scores / lexicon hits
    per_q_ratings     = []   # np.array per Q (possibly empty)
    per_q_words = []

    for qid in qids:
        q_df = df[df[qid_column] == qid]
        q_text = q_df[text_col].str.cat(sep=" ")

        if IS_SENTENCE:
            q_units = get_tokens(q_text)                     # sentences
            per_q_unit_counts.append(len(q_units))
            hits = extract_sentence_ratings(q_units, "compound")  # non-zero sentence scores
            words = []
        else:
            q_tokens, tags = get_tokens(q_text)              # words (+ tags)
            per_q_unit_counts.append(len(q_tokens))
            hits, words = extract_word_ratings(q_tokens, tags)

        r = np.array(hits, dtype=float)
        per_q_ratings.append(r)
        per_q_lex_counts.append(r.size)
        per_q_words.append(np.array(words))
    
    

    # --- Per-Q averaged metrics ---
    # keep only questions that produced at least one rating for these “*_mean of X” where X is undefined for empty
    nonempty = [r for r in per_q_ratings if r.size > 0]
    per_q_unit_counts = np.array(per_q_unit_counts)

    std_terms      = [np.std(r, ddof=1)          for r in per_q_ratings if r.size > 1]
    diff_std_terms = [np.std(np.diff(r), ddof=1) for r in per_q_ratings if r.size > 2]
    neg_mean_terms = [np.mean(r[r < 0])          for r in per_q_ratings if np.any(r < 0)]
    pos_mean_terms = [np.mean(r[r > 0])          for r in per_q_ratings if np.any(r > 0)]

    if IS_SENTENCE:
        # Set all to null
        unique_lexicon_words = []
        unique_positive_words = []
        unique_negative_words = []
        unique_response_words = []
    else:
        unique_lexicon_words = np.array([len(set(w)) for w in per_q_words])
        unique_positive_words = np.array([len(set(w[r > 0])) for r, w in zip(per_q_ratings, per_q_words)])
        unique_negative_words = np.array([len(set(w[r < 0])) for r, w in zip(per_q_ratings, per_q_words)])
        unique_response_words = np.array([len(set(w)) for w in q_tokens])

    subject_sentiment["unique_response_word_count_mean"] = float(np.mean(unique_response_words)) if len(unique_response_words) > 0 else 0
    subject_sentiment["unique_lexicon_word_count_mean"] = float(np.mean(unique_lexicon_words))   if len(unique_lexicon_words) > 0      else 0
    subject_sentiment["unique_positive_word_count_mean"] = float(np.mean(unique_positive_words)) if len(unique_positive_words) > 0 else 0
    subject_sentiment["unique_negative_word_count_mean"] = float(np.mean(unique_negative_words)) if len(unique_negative_words) > 0 else 0

    subject_sentiment["unique_lexicon_word_ratio_mean"] = float(np.mean([uw/n for uw, n in zip(unique_lexicon_words, per_q_unit_counts) if n > 0]))   if len(unique_lexicon_words) > 0      else 0
    subject_sentiment["unique_positive_word_ratio_mean"] = float(np.mean([uw/n for uw, n in zip(unique_positive_words, per_q_unit_counts) if n > 0])) if len(unique_positive_words) > 0 else 0
    subject_sentiment["unique_negative_word_ratio_mean"] = float(np.mean([uw/n for uw, n in zip(unique_negative_words, per_q_unit_counts) if n > 0])) if len(unique_negative_words) > 0 else 0

    subject_sentiment["unique_lexicon_word_ratio_rel_mean"] = float(np.mean([uw/n for uw, n in zip(unique_lexicon_words, unique_response_words) if n > 0]))   if len(unique_lexicon_words) > 0      else 0
    subject_sentiment["unique_positive_word_ratio_rel_mean"] = float(np.mean([uw/n for uw, n in zip(unique_positive_words, unique_response_words) if n > 0])) if len(unique_positive_words) > 0 else 0
    subject_sentiment["unique_negative_word_ratio_rel_mean"] = float(np.mean([uw/n for uw, n in zip(unique_negative_words, unique_response_words) if n > 0])) if len(unique_negative_words) > 0 else 0

    subject_sentiment["std_mean"]           = float(np.mean(std_terms))      if std_terms      else np.nan
    subject_sentiment["diff_std_mean"]      = float(np.mean(diff_std_terms)) if diff_std_terms else np.nan
    subject_sentiment["negative_mean_mean"] = float(np.mean(neg_mean_terms)) if neg_mean_terms else np.nan
    subject_sentiment["positive_mean_mean"] = float(np.mean(pos_mean_terms)) if pos_mean_terms else np.nan

    subject_sentiment["mean_mean"] = float(np.mean([np.mean(r) for r in nonempty])) if nonempty else np.nan
    subject_sentiment["min_mean"]  = float(np.mean([np.min(r)  for r in nonempty])) if nonempty else np.nan
    subject_sentiment["max_mean"]  = float(np.mean([np.max(r)  for r in nonempty])) if nonempty else np.nan

    # For ratios that are well-defined even when there are 0 hits, include ALL answered questions
    answered_mask = np.array(per_q_unit_counts) > 0
    per_q_unit_counts_arr = np.array(per_q_unit_counts)[answered_mask]
    per_q_lex_counts_arr  = np.array(per_q_lex_counts)[answered_mask]

    # counts of neg/pos per Q INCLUDING zeros for empty rating arrays
    per_q_neg_counts = np.array([(r < 0).sum() for r in per_q_ratings])[answered_mask]
    per_q_pos_counts = np.array([(r > 0).sum() for r in per_q_ratings])[answered_mask]

    # ratio per Q, then average
    neg_ratio_per_q = per_q_neg_counts / per_q_unit_counts_arr
    pos_ratio_per_q = per_q_pos_counts / per_q_unit_counts_arr
    lex_ratio_per_q = per_q_lex_counts_arr / per_q_unit_counts_arr

    subject_sentiment["neg_lexicon_ratio_mean"] = np.mean(np.array([(r < 0).sum()/len(r) for r in nonempty]))

    subject_sentiment["neg_ratio_mean"]          = float(np.mean(neg_ratio_per_q)) if neg_ratio_per_q.size else np.nan
    subject_sentiment["pos_ratio_mean"]          = float(np.mean(pos_ratio_per_q)) if pos_ratio_per_q.size else np.nan
    subject_sentiment["lexicon_word_ratio_mean"] = float(np.mean(lex_ratio_per_q)) if lex_ratio_per_q.size else np.nan

    # avg counts across *answered* Qs (include zeros)
    subject_sentiment["lexicon_word_count_mean"]  = float(np.mean(per_q_lex_counts_arr))  if per_q_lex_counts_arr.size  else np.nan
    subject_sentiment["response_word_count_mean"] = float(np.mean(per_q_unit_counts_arr)) if per_q_unit_counts_arr.size else np.nan

    # --- TOTAL (pooled across all answers) ---
    full_text = df[text_col].str.cat(sep=" ")

    if IS_SENTENCE:
        full_units = get_tokens(full_text)                        # sentences
        hits = extract_sentence_ratings(full_units, "compound")   # non-zero sentence scores
        lex_words = []
    else:
        full_units, tags = get_tokens(full_text)                  # words
        hits, lex_words = extract_word_ratings(full_units, tags)

    ratings     = np.array(hits, dtype=float)
    unit_count  = len(full_units)   # total sentences or words
    lex_count   = ratings.size
    pos_count = (ratings > 0).sum()
    neg_count = (ratings < 0).sum()

    subject_sentiment["lexicon_ratings"] = ratings.tolist()
    subject_sentiment["lexicon_words"]   = lex_words
    subject_sentiment["response_word_count_sum"]   = unit_count
    subject_sentiment["response_word_count_total"] = unit_count
    subject_sentiment["lexicon_word_count_total"]  = int(lex_count)

    # guarded stats
    subject_sentiment["mean_total"]      = float(np.mean(ratings))                 if lex_count else np.nan
    subject_sentiment["min_total"]       = float(np.min(ratings))                  if lex_count else np.nan
    subject_sentiment["max_total"]       = float(np.max(ratings))                  if lex_count else np.nan
    subject_sentiment["std_total"]       = float(np.std(ratings, ddof=1))          if lex_count > 1 else np.nan
    subject_sentiment["diff_std_total"]  = float(np.std(np.diff(ratings), ddof=1)) if lex_count > 2 else np.nan
    subject_sentiment["neg_lexicon_ratio_total"] = float((ratings < 0).sum() / lex_count) if lex_count else 0.0

    pos_vals = ratings[ratings > 0]
    neg_vals = ratings[ratings < 0]
    # choose NaN or 0.0 and use it consistently across your pipeline; here I match your aggregator (0.0)
    subject_sentiment["positive_mean_total"] = float(np.mean(pos_vals)) if pos_vals.size else np.nan
    subject_sentiment["negative_mean_total"] = float(np.mean(neg_vals)) if neg_vals.size else np.nan

    subject_sentiment["lexicon_word_ratio_total"] = float(lex_count / unit_count) if unit_count else 0.0
    subject_sentiment["neg_ratio_total"]          = float(neg_vals.size / unit_count) if unit_count else 0.0
    subject_sentiment["pos_ratio_total"]          = float(pos_vals.size / unit_count) if unit_count else 0.0

    lex_words = np.array(lex_words)
    if IS_SENTENCE:
        # Set all to null
        unique_lexicon_words = 0
        unique_positive_words = 0
        unique_negative_words = 0
        unique_response_words = 0
    else:
        unique_lexicon_words = len(set(lex_words))
        unique_positive_words = len(set(np.array(lex_words)[ratings > 0].tolist()))
        unique_negative_words = len(set(np.array(lex_words)[ratings < 0].tolist()))
        unique_response_words = len(set(full_units)) # General Vocabulary Size

    subject_sentiment["unique_response_word_count_total"] = unique_response_words

    subject_sentiment["unique_lexicon_word_count_total"] = unique_lexicon_words
    subject_sentiment["unique_positive_word_count_total"] = unique_positive_words
    subject_sentiment["unique_negative_word_count_total"] = unique_negative_words

    subject_sentiment["unique_lexicon_word_ratio_total"] = float(unique_lexicon_words / unit_count)   if unit_count else 0
    subject_sentiment["unique_positive_word_ratio_total"] = float(unique_positive_words / unit_count) if unit_count else 0
    subject_sentiment["unique_negative_word_ratio_total"] = float(unique_negative_words / unit_count) if unit_count else 0

    subject_sentiment["unique_lexicon_word_ratio_rel_total"] = float(unique_lexicon_words / unique_response_words)   if unique_response_words else 0
    subject_sentiment["unique_positive_word_ratio_rel_total"] = float(unique_positive_words / unique_response_words) if unique_response_words else 0
    subject_sentiment["unique_negative_word_ratio_rel_total"] = float(unique_negative_words / unique_response_words) if unique_response_words else 0

    subject_sentiment["unique_positive_to_positive_word_ratio_total"] = float(unique_positive_words / pos_count) if pos_count else 0
    subject_sentiment["unique_negative_to_negative_word_ratio_rel_total"] = float(unique_negative_words / neg_count) if neg_count else 0


    return subject_sentiment


import traceback 
if __name__ == "__main__":
    # Get all subjects with verified transcripts and process them
    subjects = sorted(
    get_subject_id(f) for f in os.listdir(TEXT_DIR) if f.endswith("_qid.xlsx")
    )   
    print(subjects)
    print("Total subjects:", len(subjects))

    results = []
    for i, subject_id in enumerate(subjects):
        print(f"\n[{i}/{len(subjects)}] → starting {subject_id}")
        try:
            result = process_subject(subject_id)
            results.append(result)
            print(f"[{subject_id}] ✓ finished\n")
        except Exception as e:
            print(f"[{subject_id}] ✗ {e}")
            traceback.print_exc()
    df = pd.DataFrame(results)
    df.to_excel(RESULTS_DIR / f"sentiment_{test_id}.xlsx", index=False, engine="openpyxl")

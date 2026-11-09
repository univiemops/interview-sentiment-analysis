**Sentiment Analysis with LIWC and VADER**

Correlational analysis of sentiment-analysis features and clinical ratings of identity pathology (STIPO). This is the supplementary code to the paper *Sentiment Analysis of Clinical Interviews Identifies Language Markers of Identity Diffusion in Adolescents with Personality Disorder*: https://osf.io/preprints/psyarxiv/dqbz2_v1.

## Workflow

1. **Raw transcripts**: `data/transcripts_raw/`, one Excel file per subject (`T0001.xlsx`, ...).
2. **Question labelling**: `src/question_labels/`. `question_labels_fuzzy.py` (fuzzy matching) or `question_labels_sbert.py` (SBERT semantic matching) label each response and write candidates to `data/question_labels/to_verify/`. Review them by hand and save the results to `data/question_labels/verified/`. For a new batch, update `questions_dict.py` before labelling it.
3. **Identity section export**: `src/question_labels/export_verified_qid.ipynb` keeps the relevant questions and responses from the verified files and writes the German transcripts to `data/transcripts_german/`.
4. **Translation**: `src/translation/llm_translation.py` translates the German transcripts into English with a GPT model, using the prompt in `translation_prompt.json`. Output goes to `data/transcripts/`. Interrupted runs can be resumed.
5. **Sentiment analysis**: `src/sentiment/sentiment_analysis.py` scores the transcripts with VADER (English translations), GerVADER and SentiWS (German text), at word and sentence level. Output: `data/results/sentiment_long.xlsx`.
6. **LIWC analysis**: the texts are scored in the LIWC software, LIWC-22 for English and DE-LIWC2015 for German (see the [LIWC website](https://www.liwc.app/)). `data/liwc_results/liwc_create.ipynb` prepares the texts for LIWC, and `liwc_merge.ipynb` combines the LIWC output into `data/liwc_results/liwc_long.xlsx`.
7. **Statistical analysis**: `src/analysis/analysis_final.ipynb` joins the STIPO clinician scores (`data/patient_data/stipo.xlsx`) with the sentiment and LIWC features, checks data fidelity per participant, and computes the correlations with Bonferroni correction. Output: `data/results/correlations.xlsx` and `data/results/features_to_correlate.xlsx`.
8. **Figures**: `src/analysis/scatterplots.ipynb` plots the manuscript figures, the German vs English comparisons, and the per-measure pair plots from `features_to_correlate.xlsx`. All figures are saved to `data/results/figures/`.

## Setup

Dependencies are listed in `requirements.txt`.

## Lexicons

The lexicon files are not in this repository. Download them and place them in `data/lexicons/` under the names below before running `src/sentiment/sentiment_analysis.py`. Word-level scoring needs all three. Sentence-level VADER scoring uses the VADER package's built-in analyzer and needs no file.

| File in `data/lexicons/` | Used for | Where to get it |
|---|---|---|
| `vader_lexicon.txt` | VADER, English | [cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment) (Hutto & Gilbert, 2014) |
| `GERVaderLexicon.txt` | GerVADER, German | [KarstenAMF/GerVADER](https://github.com/KarstenAMF/GerVADER) (Tymann et al., 2019) |
| `sentiws.csv` | SentiWS, German | [Leipzig Wortschatz downloads](https://wortschatz.uni-leipzig.de/en/download) (SentiWS v2.0; Remus, Quasthoff & Heyer, 2010) |

The LIWC dictionaries are licensed, so they are not available from this repository either. The LIWC output files used in step 6 come from the LIWC software ([liwc.app](https://www.liwc.app/)).

## Data note

Everything in `data/` is **simulated example data**. The transcripts T0001–T0005 were generated via GPT using the `interview_simulation_prompt.txt` so that a pipeline that runs end-to-end and the code can be shared via git. None of it is real participant data.

## References

Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216–225. https://doi.org/10.1609/icwsm.v8i1.14550

Meier, T., Boyd, R. L., Pennebaker, J. W., Mehl, M. R., Martin, M., Wolf, M., & Horn, A. B. (2019). "LIWC auf Deutsch": The Development, Psychometrics, and Introduction of DE-LIWC2015. *PsyArXiv*. https://doi.org/10.31234/osf.io/uq8zt

OpenAI. (2024). *Hello GPT-4o*. https://openai.com/index/hello-gpt-4o/

Pennebaker, J. W., Boyd, R. L., Jordan, K., & Blackburn, K. (2015). *The Development and Psychometric Properties of LIWC2015*. http://hdl.handle.net/2152/31333

Tymann, K. M., Lutz, M., Palsbröker, P., & Gips, C. (2019). *GerVADER: A German adaptation of the VADER sentiment analysis tool for social media texts*.

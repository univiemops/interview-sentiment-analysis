# -*- coding: utf-8 -*-
"""
Translate German patient snippets to English, one interview at a time.

Implementation notes:
- Uses the OpenAI Responses API (client.responses.create) pointed at the Azure
  endpoint given in the credentials file (OpenAI client with base_url=endpoint).
- Requires an API key and a deployment name, read from the credentials file at
  ~/Azure/credentials-gpt-5-mini.txt (plain key=value lines: endpoint, api_key,
  deployment_name).
- Only patient turns are translated. Each call gets the 3 previous turns
  (either speaker) as context, so the model can follow the conversation.
- store=False: Azure does not keep the response on the server.
- Input:  data/transcripts_german/<subject>_qid.xlsx
  Output: data/transcripts/<subject>_qid.xlsx
  (all original columns plus an "english_text" column).
- The output is saved after every call. Rows that already have a translation
  in the output file are skipped, so an interrupted run can be resumed.
- Progress, the full prompt and response, and token usage are printed to the
  terminal.
"""
import os, time, json
import pandas as pd
import openai
from pathlib import Path
from collections import Counter, deque
import traceback

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
CREDS = Path("~/Azure/credentials-gpt-5-mini.txt").expanduser()  # outside the project, see README
INPUT_DIR = ROOT_DIR / "data" / "transcripts_german"
OUTPUT_DIR = ROOT_DIR / "data" / "transcripts"
PROMPT_FILE = CURRENT_DIR / "translation_prompt.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # create the output folder on import, if missing


def load_credentials(path):
    """Read key=value lines from the credentials file into a dict.

    Blank lines, lines without "=", and lines starting with "#" are skipped.
    Surrounding quotes and whitespace are removed from both keys and values.
    """
    with open(path) as f:
        return {
            key.strip(): value.strip().strip('"').strip("'")
            for line in f
            if "=" in line and not line.strip().startswith("#")
            for key, value in [line.split("=", 1)]
        }


# ---------------------------------------------------------------------------
# Terminal output helpers. These only format the log; they do not affect the
# translation.
# ---------------------------------------------------------------------------
WIDTH = 88  # width of the horizontal rules

def rule(char="─"):
    """Print a horizontal line made of char."""
    print(char * WIDTH)

def section(title):
    """Print a double-ruled heading, e.g. at the start of a run or subject."""
    print()
    rule("═")
    print(f"  {title}")
    rule("═")

def block(label, text):
    """Print a labelled, indented block of text with its character count."""
    print(f"  {label} ({len(text)} chars):")
    for line in text.splitlines() or [""]:
        print(f"    {line}")
# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def call_model(client, deployment, instructions, prompt):
    """Send one prompt to the model and print the input, output and token counts.

    Returns (translation, tokens_in, tokens_out). Errors are not caught here;
    the caller handles them.
    """
    block("Input", prompt)
    t_call = time.time()
    response = client.responses.create(
        model=deployment,  # Azure deployment name
        instructions=instructions,
        input=prompt,
        store=False,  # do not keep the response on the server
        #temperature=0,
    )
    translation = response.output_text.strip()
    tokens_in, tokens_out = response.usage.input_tokens, response.usage.output_tokens
    block("Output", translation)
    print(f"  tokens in={tokens_in} out={tokens_out} · {time.time() - t_call:.1f}s")
    return translation, tokens_in, tokens_out


def translate_subject(file, client, deployment, instructions, prompts, totals):
    """Translate the patient turns of one subject and save the result.

    Adds this subject's calls and tokens to totals (the run-wide Counter) and
    prints a one-line summary when it finishes.
    """
    t0 = time.time()
    subject_id = file.stem.removesuffix("_qid")
    file_totals = Counter()  # counts for this subject only
    try:
        df = pd.read_excel(file, dtype=str)
        df["text_english"] = None
        context = deque(maxlen=3)  # the 3 most recent turns, both speakers
        n_patient = (df["role"] == "Patient:").sum()

        # Resume: if a previous run already wrote this subject's output, reuse
        # its translations. Only do so if the rows match the source exactly.
        out_path = OUTPUT_DIR / f"{subject_id}_qid.xlsx"
        if out_path.exists():
            done = pd.read_excel(out_path, dtype=str)
            if len(done) == len(df) and (done["text"] == df["text"]).all():
                df["text_english"] = done["text_english"]
                print(f"  resuming from existing {out_path.name}")
            else:
                print(f"  {out_path.name} does not match the source file - translating from scratch")
        section(f"{subject_id}  ({file.name}, {n_patient} patient turns)")

        for idx, row in df.iterrows():
            role, text = row["role"], row["text"]

            # Already translated in an earlier run: no API call.
            if role == "Patient:" and pd.notna(df.at[idx, "text_english"]):
                print(f"\n── row {idx} · {row['qid']} · already translated, skipping")
                context.append({"role": role, "text": text})
                continue

            if role == "Patient:":
                # Fill the prompt template: [CONTEXT] = previous turns, [SNIPPET] = this turn.
                context_text = "\n".join(f"{x['role']} {x['text']}" for x in context)
                prompt = prompts["user_prompt"].replace("[CONTEXT]", context_text).replace("[SNIPPET]", text)
                print(f"\n── row {idx} · {row['qid']} " + "─" * 60)
                try:
                    translation, tokens_in, tokens_out = call_model(client, deployment, instructions, prompt)
                    df.at[idx, "text_english"] = translation
                    for stats in (file_totals, totals):  # update subject and run totals
                        stats["calls"] += 1
                        stats["in"] += tokens_in
                        stats["out"] += tokens_out
                except Exception as e:
                        file_totals["errors"] += 1
                        totals["errors"] += 1
                        print(f"  ERROR on row {idx}: {e}")

                # Save after every patient turn, so progress is kept if the run stops.
                df.to_excel(out_path, index=False)

            context.append({"role": role, "text": text})

        df.to_excel(out_path, index=False)  # final save, covers subjects with no patient turns
        print(f"\n  saved → {out_path}")
    except Exception as e:
        # Errors outside the per-row calls, e.g. unreadable file or a save failure.
        file_totals["errors"] += 1
        totals["errors"] += 1
        print(f"  ERROR processing {subject_id}: {e}")

    print(f"  {subject_id} done in {time.time() - t0:.1f}s · calls={file_totals['calls']} · "
          f"tokens in={file_totals['in']} out={file_totals['out']} · errors={file_totals['errors']}")


def main():
    creds = load_credentials(CREDS)
    with open(PROMPT_FILE, encoding="utf-8") as f:
        prompts = json.load(f)  # keys used: system_prompt, user_prompt

    client = openai.OpenAI(
        base_url=creds["endpoint"],
        api_key=creds["api_key"],
    )
    instructions = prompts["system_prompt"]  # sent as the Responses API "instructions"
    totals = Counter()  # run-wide counts: calls, in, out, errors

    section("Translation run")
    print(f"  model: {creds['deployment_name']}")  # the deployment name is not secret
    block("System instructions", instructions)

    for file in INPUT_DIR.glob("*_qid.xlsx"):
        translate_subject(file, client, creds["deployment_name"], instructions, prompts, totals)

    section("Run summary")
    print(f"  calls={totals['calls']} · tokens in={totals['in']} out={totals['out']} "
          f"(total={totals['in'] + totals['out']}) · errors={totals['errors']}")
    rule("═")


if __name__ == "__main__":
    main()

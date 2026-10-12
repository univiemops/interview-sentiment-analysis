# -*- coding: utf-8 -*-
"""Translate German patient snippets to English via Azure OpenAI with 2-turn context."""
import os, re, time, json
import pandas as pd
from openai import AzureOpenAI
from pathlib import Path
from collections import deque

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
INPUT_DIR = ROOT_DIR / "data" / "transcripts_german"
OUTPUT_DIR = ROOT_DIR / "data" / "transcripts"
PROMPT_FILE = CURRENT_DIR / "translation_prompt.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_ID = "gpt-4o"  # Azure OpenAI model ID for translation
def load_credentials(path):
    with open(path) as f:
        return dict(line.strip().split("=", 1) for line in f if "=" in line)

def main():
    creds = load_credentials(ROOT_DIR / "credentials.txt")
    with open(PROMPT_FILE, encoding="utf-8") as f:
        prompts = json.load(f)

    client = AzureOpenAI(
        azure_endpoint=creds["azure_endpoint"],
        api_version=creds["api_version"],
        api_key=creds["api_key"],
    )
    system_msg = {"role": "system", "content": prompts["system_prompt"]}

    for file in INPUT_DIR.glob("*_qid.xlsx"):
        t0 = time.time()
        subject_id = file.stem.removesuffix("_qid")
        print(f"Processing {subject_id}...")
        try:
            df = pd.read_excel(file, dtype=str)[["role", "text", "qid"]]
            df["text_english"] = None
            context = deque(maxlen=2)

            for idx, row in df.iterrows():
                role, text = row["role"], row["text"]

                if role == "Patient":
                    context_text = "\n".join(f"{x['role']}: {x['text']}" for x in context)
                    prompt = prompts["user_prompt"].replace("[CONTEXT]", context_text).replace("[SNIPPET]", text)
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_ID,
                            messages=[system_msg, {"role": "user", "content": prompt}],
                            #temperature=0,
                        )
                        df.at[idx, "text_english"] = response.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"↳ error on row {idx}: {e}")

                context.append({"role": role, "text": text})

            df.to_excel(OUTPUT_DIR / file.name, index=False)
            print(f"Processed {subject_id} in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")

if __name__ == "__main__":
    main()
"""
preprocess_botsv3.py — Preprocess the labeled BOTS v3 combined dataset
for SecureBERT fine-tuning.

Design: Uses a format adapter pattern so the same pipeline works with
BOTS v3 data now and Wazuh alert JSON later — just swap the adapter.

Usage:
    python preprocess_botsv3.py

Input:  data/botsv3_combined.csv (from combine_botsv3_v2.py)
Output: data/botsv3_train.csv, data/botsv3_test.csv
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH = "data/botsv3_combined.csv"
TRAIN_PATH = "data/botsv3_train.csv"
TEST_PATH = "data/botsv3_test.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TEXT_LEN = 512  # SecureBERT max token input is 512


# ── FORMAT ADAPTERS ───────────────────────────────────────────────────────────
# Swap this function when switching data sources. The fine-tuning script
# only sees the output: a clean text string + a binary label.

def adapt_botsv3(row):
    """
    BOTS v3 adapter — alert_text is already natural language.
    Prepend source_category as a prefix so the model learns to distinguish
    alert types. This prefix pattern carries over to Wazuh.
    """
    category = row.get("source_category", "unknown")
    text = row.get("alert_text", "")
    return f"[{category}] {text}"


def adapt_wazuh(alert_json):
    """
    Wazuh adapter — extracts and formats fields from a Wazuh JSON alert.
    Source: /var/ossec/logs/alerts/alerts.json or Wazuh API.
    Filters: level >= 4 recommended before calling this function.
    """
    rule = alert_json.get("rule", {})

    # Skip alerts below level 4
    level = rule.get("level", 0)
    if level < 4:
        return None

    agent = alert_json.get("agent", {})
    data = alert_json.get("data", {})

    parts = []

    # Category from rule groups
    groups = rule.get("groups", [])
    if groups:
        parts.append(f"[{','.join(groups)}]")

    # Rule description — primary analyst-facing text
    desc = rule.get("description", "")
    if desc:
        parts.append(desc)

    if level:
        parts.append(f"level={level}")

    # Agent context
    agent_name = agent.get("name", "")
    if agent_name:
        parts.append(f"agent={agent_name}")

    # Parsed data fields (srcip, protocol, url, etc.)
    srcip = data.get("srcip", "")
    if srcip:
        parts.append(f"src={srcip}")
    protocol = data.get("protocol", "")
    if protocol:
        parts.append(f"proto={protocol}")
    url = data.get("url", "")
    if url:
        parts.append(f"url={url}")

    # Raw log as fallback context
    full_log = alert_json.get("full_log", "")
    if full_log:
        parts.append(f"| {full_log}")

    return " ".join(parts)


# ── TEXT CLEANING ─────────────────────────────────────────────────────────────

def clean_text(text):
    """
    Normalize alert text for SecureBERT consumption.
    Keeps security-relevant tokens (IPs, ports, paths, EventCodes).
    """
    if not isinstance(text, str):
        return ""

    # Collapse excessive whitespace and tabs (common in Windows event logs)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'\n+', ' | ', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove null bytes and control characters (keep printable + basic whitespace)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Truncate to MAX_TEXT_LEN chars (SecureBERT tokenizer handles the rest)
    text = text[:MAX_TEXT_LEN * 4]  # rough char estimate, tokenizer does final cut

    return text.strip()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded: {len(df)} rows")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nSource categories:\n{df['source_category'].value_counts()}")

    # Apply format adapter + cleaning
    print("\nApplying BOTS v3 format adapter...")
    df["text"] = df.apply(adapt_botsv3, axis=1).apply(clean_text)

    # Convert labels to binary integers (threat=1, benign=0)
    df["label_int"] = (df["label"] == "threat").astype(int)

    # Drop rows with empty text after cleaning
    before = len(df)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with empty/short text")

    # Stratified train/test split — stratify by BOTH label and source_category
    # to ensure each split has representative samples from all categories
    df["strat_key"] = df["label"] + "_" + df["source_category"]

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["strat_key"],
    )

    # Save only the columns the fine-tuning script needs
    train_out = train_df[["text", "label_int", "label", "source_category"]].copy()
    test_out = test_df[["text", "label_int", "label", "source_category"]].copy()

    train_out.to_csv(TRAIN_PATH, index=False)
    test_out.to_csv(TEST_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Train set: {len(train_out)} rows → {TRAIN_PATH}")
    print(f"Test set:  {len(test_out)} rows → {TEST_PATH}")
    print(f"\nTrain label distribution:")
    print(f"  Threat: {(train_out['label_int']==1).sum()} ({100*(train_out['label_int']==1).mean():.1f}%)")
    print(f"  Benign: {(train_out['label_int']==0).sum()} ({100*(train_out['label_int']==0).mean():.1f}%)")
    print(f"\nTrain source_category distribution:")
    print(train_out["source_category"].value_counts().to_string())
    print(f"\nSample processed text:")
    for _, row in train_out.head(3).iterrows():
        print(f"  [{row['label']}] {row['text'][:120]}...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
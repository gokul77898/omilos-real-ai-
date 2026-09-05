#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "legal_clean"
RAW_JSONL = RAW_DIR / "open_indian_legal.jsonl"

TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized_128k_32k"
MANIFEST = TOKENIZED_DIR / "corpus_manifest.json"

HF_DATASET = "OmilosAISolutions/omilos-indian-legal-pretrain-v1"


def run(cmd):
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(map(str, cmd)))
    print("=" * 80)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def verify():
    if not RAW_JSONL.exists():
        raise RuntimeError(f"Missing cleaned corpus: {RAW_JSONL}")

    if not MANIFEST.exists():
        raise RuntimeError(f"Missing corpus manifest: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text())

    if manifest.get("accepted_documents", 0) <= 0:
        raise RuntimeError("No accepted documents.")

    if manifest.get("packing", {}).get("train", {}).get("finalized") is not True:
        raise RuntimeError("Train corpus is not finalized.")

    if manifest.get("packing", {}).get("validation", {}).get("finalized") is not True:
        raise RuntimeError("Validation corpus is not finalized.")

    if manifest.get("sequence_length") != 131072:
        raise RuntimeError(
            f"Wrong sequence length: {manifest.get('sequence_length')}"
        )

    tokenizer = manifest.get("tokenizer", {})
    if tokenizer.get("vocab_size") != 32000:
        raise RuntimeError("Wrong tokenizer vocabulary size.")

    special = tokenizer.get("special_token_ids", {})
    expected = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
    if special != expected:
        raise RuntimeError(f"Wrong special-token IDs: {special}")

    train_dir = TOKENIZED_DIR / "train"
    val_dir = TOKENIZED_DIR / "validation"

    train_bins = list(train_dir.glob("*.bin"))
    val_bins = list(val_dir.glob("*.bin"))

    if not train_bins:
        raise RuntimeError("No training shards found.")

    if not val_bins:
        raise RuntimeError("No validation shards found.")

    print("\n" + "=" * 80)
    print("CORPUS VERIFICATION PASSED")
    print("=" * 80)
    print("Documents:", manifest["accepted_documents"])
    print("Sequence length:", manifest["sequence_length"])
    print("Train sequences:",
          manifest["packing"]["train"]["packed_sequences"])
    print("Validation sequences:",
          manifest["packing"]["validation"]["packed_sequences"])
    print("Train tokens:",
          manifest["packing"]["train"]["total_packed_tokens"])
    print("Validation tokens:",
          manifest["packing"]["validation"]["total_packed_tokens"])
    print("Train shards:", len(train_bins))
    print("Validation shards:", len(val_bins))


def upload():
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Refusing to upload without authentication."
        )

    api = HfApi(token=token)

    print("\n" + "=" * 80)
    print("UPLOADING VERIFIED CORPUS")
    print("=" * 80)
    print("Destination:", HF_DATASET)

    api.create_repo(
        repo_id=HF_DATASET,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    # Upload the cleaned evidence-bearing JSONL + source manifest.
    api.upload_folder(
        repo_id=HF_DATASET,
        repo_type="dataset",
        folder_path=str(RAW_DIR),
        path_in_repo="legal_clean",
        commit_message="Upload cleaned open Indian legal corpus",
    )

    # Upload the verified 128K tokenized corpus.
    api.upload_folder(
        repo_id=HF_DATASET,
        repo_type="dataset",
        folder_path=str(TOKENIZED_DIR),
        path_in_repo="tokenized_128k_32k",
        commit_message="Upload verified 32K 128K-token training corpus",
    )

    print("\n" + "=" * 80)
    print("HF UPLOAD COMPLETE")
    print("=" * 80)
    print(f"https://huggingface.co/datasets/{HF_DATASET}")


def main():
    print("=" * 80)
    print("FULL OPEN INDIAN LEGAL CORPUS PIPELINE")
    print("=" * 80)
    print("GATED DATASETS: NONE")
    print("DATASET: Overthelex")
    print("TEXT: full_text")
    print("TOKENIZER: 32K")
    print("SEQUENCE: 131072")
    print("HF DESTINATION:", HF_DATASET)

    # Step 1: Download + clean + deduplicate.
    run([
        sys.executable,
        "scripts/prepare_legal_corpus.py",
    ])

    # Step 2: Tokenize + pack into 128K sequences.
    run([
        sys.executable,
        "scripts/build_long_context_corpus.py",
        "--input-jsonl",
        str(RAW_JSONL),
        "--output-dir",
        str(TOKENIZED_DIR),
        "--seq-len",
        "131072",
        "--validation-percent",
        "5",
    ])

    # Step 3: Verify EVERYTHING before upload.
    verify()

    # Step 4: Only now upload.
    upload()


if __name__ == "__main__":
    main()

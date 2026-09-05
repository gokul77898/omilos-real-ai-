#!/usr/bin/env python3
"""Build canonical 32K-token, long-context shards from an evidence-bearing JSONL file."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.ingestion import build_corpus, read_jsonl
from src.tokenizer import LegalTokenizer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Locally supplied, provenance-bearing source manifest")
    parser.add_argument("--output-dir", default="data/tokenized_128k_32k")
    parser.add_argument("--seq-len", type=int, default=131072)
    parser.add_argument("--validation-percent", type=int, default=5)
    args = parser.parse_args()
    tokenizer = LegalTokenizer.load(PROJECT_ROOT / "artifacts/tokenizer_32k")
    manifest = build_corpus(read_jsonl(args.input_jsonl), tokenizer, args.output_dir, args.seq_len, args.validation_percent)
    print(f"CORPUS VERIFIED: {manifest['accepted_documents']} documents; manifest={Path(args.output_dir) / 'corpus_manifest.json'}")

if __name__ == "__main__":
    main()

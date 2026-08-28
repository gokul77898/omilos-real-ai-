from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

DATA_DIR = Path("data/tokenizer_train")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("overthelex/indian-court-decisions", "supreme_court", ["full_text"], "supreme_court.txt"),
    ("KanoonGPT/indian-legal-documents", None, ["text"], "legal_docs.txt"),
    ("123Divyansh/Constitution.-of-India-TXT_File", None, ["text"], "constitution.txt"),
    ("navaneeth005/BNS_detailed", None, ["Section", "Offence", "Punishment"], "bns_detailed.txt"),
    ("navaneeth005/BNS_definitions", None, ["Section", "Title", "Legal Definition"], "bns_defs.txt"),
    ("SnehaDeshmukh/IndianBailJudgments-1200", None, ["facts", "judgment_reason", "legal_principles_discussed"], "bail.txt"),
]

for name, config, fields, fname in SOURCES:
    out = DATA_DIR / fname
    if out.exists():
        print(f"SKIP (exists): {fname}")
        continue
    print(f"Downloading: {fname}", flush=True)
    try:
        ds = load_dataset(name, config, streaming=True) if config else load_dataset(name, streaming=True)
        with open(out, "w", encoding="utf-8") as f:
            count = 0
            for split, stream in ds.items():
                for row in stream:
                    parts = [str(row[f]) for f in fields if row.get(f)]
                    text = "\n".join(parts).strip()
                    if text:
                        f.write(text + "\n")
                        count += 1
                        if count % 5000 == 0:
                            print(f"  {fname}: {count} docs", flush=True)
        print(f"DONE: {fname} ({count} docs)", flush=True)
    except Exception as e:
        print(f"SKIPPED {fname}: {e}", flush=True)

files = sorted(DATA_DIR.glob("*.txt"))
print(f"\nTraining 32K tokenizer on {len(files)} files...", flush=True)
for f in files:
    print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[str(f) for f in files],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
)

out_dir = Path("artifacts/tokenizer_32k")
out_dir.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(out_dir / "tokenizer.json"))
print(f"\nDone! vocab_size=32000")
print(f"Saved to {out_dir}")

from pathlib import Path
from datasets import load_dataset
from src.tokenizer import LegalTokenizer
from src.config import TokenizerConfig

SOURCES = [
    ("overthelex/indian-court-decisions", "high_courts", ["full_text"]),
    ("overthelex/indian-court-decisions", "supreme_court", ["full_text"]),
    ("KanoonGPT/indian-case-laws", None, ["indexable_text"]),
    ("KanoonGPT/indian-legal-documents", None, ["text"]),
    ("123Divyansh/Constitution.-of-India-TXT_File", None, ["text"]),
    ("navaneeth005/BNS_detailed", None, ["Section", "Offence", "Punishment"]),
    ("navaneeth005/BNS_definitions", None, ["Section", "Title", "Legal Definition"]),
    ("SnehaDeshmukh/IndianBailJudgments-1200", None, ["facts", "judgment_reason", "legal_principles_discussed"]),
]

OUTPUT = Path("artifacts/tokenizer_32k")

def text_iter():
    for name, config, fields in SOURCES:
        label = f"{name}:{config}" if config else name
        print(f"Streaming: {label}", flush=True)
        try:
            ds = load_dataset(name, config, streaming=True) if config else load_dataset(name, streaming=True)
            for split, stream in ds.items():
                for row in stream:
                    parts = [str(row[f]) for f in fields if row.get(f)]
                    text = "\n".join(parts).strip()
                    if text:
                        yield text
        except Exception as e:
            print(f"SKIPPED {label}: {e}", flush=True)

tokenizer = LegalTokenizer(config=TokenizerConfig(vocab_size=32000, min_frequency=2))
print("Training 32K tokenizer...", flush=True)
tokenizer.train_from_iterator(text_iter(), vocab_size=32000, min_frequency=2)
tokenizer.save(OUTPUT)
print(f"Done. vocab_size={tokenizer.vocab_size}", flush=True)
print(f"Saved to {OUTPUT}", flush=True)

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.model import LegalCausalLM
from src.seed import set_seed
from src.trainer import Trainer
from src.tokenizer import LegalTokenizer


class LegalSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.records = []

        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                messages = record["messages"]

                user = messages[0]["content"].strip()
                assistant = messages[1]["content"].strip()

                if not user or not assistant:
                    continue

                self.records.append(
                    {
                        "id": record["id"],
                        "user": user,
                        "assistant": assistant,
                    }
                )

        if not self.records:
            raise ValueError(f"No usable SFT records in {self.path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        prompt = (
            "Question: "
            + record["user"]
            + "\nAnswer: "
        )

        full_text = prompt + record["assistant"]

        encoded = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
        )

        full_ids = encoded.ids

        # Tokenize the prompt separately so loss is applied only to
        # the assistant response.
        prompt_encoded = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        prompt_len = len(prompt_encoded.ids)

        if prompt_len >= self.max_seq_len:
            raise ValueError(
                f"Prompt alone exceeds max_seq_len: {record['id']}"
            )

        ids = full_ids[: self.max_seq_len]

        if len(ids) <= prompt_len:
            raise ValueError(
                f"Answer truncated away: {record['id']}"
            )

        input_ids = torch.tensor(ids, dtype=torch.long)

        labels = input_ids.clone()

        # Ignore prompt tokens in the loss.
        labels[:prompt_len] = -100

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def collate_batch(batch, pad_token_id):
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids = []
    labels = []
    attention_masks = []

    for item in batch:
        length = item["input_ids"].shape[0]
        pad_len = max_len - length

        input_ids.append(
            torch.cat(
                [
                    item["input_ids"],
                    torch.full(
                        (pad_len,),
                        pad_token_id,
                        dtype=torch.long,
                    ),
                ]
            )
        )

        labels.append(
            torch.cat(
                [
                    item["labels"],
                    torch.full(
                        (pad_len,),
                        -100,
                        dtype=torch.long,
                    ),
                ]
            )
        )

        attention_masks.append(
            torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros(
                        pad_len,
                        dtype=torch.long,
                    ),
                ]
            )
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks),
    }


def load_native_checkpoint(model, checkpoint):
    checkpoint = Path(checkpoint)

    if checkpoint.is_dir():
        model_file = checkpoint / "model.pt"
    else:
        model_file = checkpoint

    if not model_file.exists():
        raise FileNotFoundError(model_file)

    state = torch.load(
        model_file,
        map_location="cpu",
        weights_only=True,
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}"
        )

    print(
        f"Loaded checkpoint: {model_file}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/500m.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--validation",
        required=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    set_seed(config.project.seed)

    print("Loading tokenizer...", flush=True)

    tokenizer = LegalTokenizer.load(
        "artifacts/tokenizer_32k"
    )

    print("Tokenizer vocab:", tokenizer.vocab_size, flush=True)

    print("Building model...", flush=True)

    model = LegalCausalLM(config.model)

    print("Loading 20K checkpoint...", flush=True)

    load_native_checkpoint(
        model,
        args.checkpoint,
    )

    print("Creating train dataset...", flush=True)

    train_dataset = LegalSFTDataset(
        args.train,
        tokenizer,
        config.model.max_seq_len,
    )

    print(
        "Train examples:",
        len(train_dataset),
        flush=True,
    )

    print("Creating validation dataset...", flush=True)

    validation_dataset = LegalSFTDataset(
        args.validation,
        tokenizer,
        config.model.max_seq_len,
    )

    print(
        "Validation examples:",
        len(validation_dataset),
        flush=True,
    )

    collate = lambda batch: collate_batch(
        batch,
        tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=validation_loader,
        device=args.device,
    )

    print("Starting SFT pilot...", flush=True)

    trainer.train(
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

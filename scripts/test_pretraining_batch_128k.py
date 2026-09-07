#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.sharding import ShardedDataset
from src.model import LegalCausalLM


DATA_ROOT = Path.home() / (
    ".cache/huggingface/hub/"
    "datasets--OmilosAISolutions--omilos-indian-legal-pretrain-v1/"
    "snapshots/3a128dd391359e17d448ab1aa18c75f258a7c321/"
    "tokenized_128k_32k"
)


def main() -> None:
    print("=" * 80)
    print("REAL 128K GPU PRETRAINING BATCH TEST")
    print("=" * 80)

    assert torch.cuda.is_available(), "CUDA is not available"

    device = torch.device("cuda")
    config = load_config(PROJECT_ROOT / "configs/500m.yaml")
    mc = config.model

    assert mc.vocab_size == 32000
    assert mc.max_seq_len == 131072
    assert mc.attention_window == 8192
    assert mc.attention_chunk_size == 1024

    manifest_path = DATA_ROOT / "corpus_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    manifest = json.loads(manifest_path.read_text())

    assert manifest["sequence_length"] == 131072
    assert manifest["tokenizer"]["vocab_size"] == 32000
    assert manifest["packing"]["train"]["packed_sequences"] == 26316

    train_dir = DATA_ROOT / "train"

    dataset = ShardedDataset(
        train_dir,
        seq_len=131072,
        vocab_size=32000,
        reject_pad=True,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset sequences: {len(dataset):,}")
    print(f"Train shards: {len(dataset.bin_files)}")

    batch = next(iter(DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )))

    input_ids = batch["input_ids"].to(device=device, non_blocking=True)
    labels = batch["labels"].to(device=device, non_blocking=True)

    print("Batch shape:", tuple(input_ids.shape))
    print("dtype:", input_ids.dtype)
    print("min token:", int(input_ids.min()))
    print("max token:", int(input_ids.max()))
    print("PAD count:", int((input_ids == 0).sum()))

    assert input_ids.shape == (1, 131072)
    assert not torch.any(input_ids == 0)
    assert int(input_ids.max()) < 32000

    print("\nLoading model...")
    model = LegalCausalLM(mc).to(
        device=device,
        dtype=torch.bfloat16,
    ).train()

    model.gradient_checkpointing_enable()

    params = sum(p.numel() for p in model.parameters())
    print("Parameters:", f"{params:,}")

    assert params == 482_827_392

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nFORWARD + BACKWARD TEST")
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    output = model(
        input_ids=input_ids,
        labels=labels,
        return_logits=False,
    )

    loss = output.loss

    print("Loss:", loss.detach().item())

    assert torch.isfinite(loss), "Loss is not finite"

    loss.backward()

    grad_tensors = 0
    bad_grads = 0

    for p in model.parameters():
        if p.grad is not None:
            grad_tensors += 1
            if not torch.isfinite(p.grad).all():
                bad_grads += 1

    assert grad_tensors > 0
    assert bad_grads == 0

    optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print("Gradient tensors:", grad_tensors)
    print("Bad gradients:", bad_grads)
    print("Optimizer step: PASS")
    print(f"Elapsed: {elapsed:.2f} sec")
    print(f"Peak allocated VRAM: {peak_gb:.2f} GB")

    print("\n" + "=" * 80)
    print("128K GPU TRAINING TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()

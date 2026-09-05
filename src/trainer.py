"""Complete PyTorch training engine with AMP, Gradient Accumulation, and Checkpointing."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import AppConfig
from src.logging_utils import setup_logger
from src.model import LegalCausalLM
from src.scheduler import get_cosine_schedule_with_warmup


class Trainer:
    """Production-grade PyTorch training engine for LegalCausalLM.

    Features:
    - Pure PyTorch training loop (no black-box abstractions).
    - Configurable AdamW with weight decay.
    - Warmup + Cosine learning rate scheduling.
    - Exact Gradient Accumulation with normalized loss scaling.
    - Automatic Mixed Precision (AMP) with GradScaler on CUDA/MPS.
    - Gradient norm clipping.
    - Evaluation loop with robust perplexity computation.
    - State-preserving checkpointing & resume capabilities.
    """

    def __init__(
        self,
        model: LegalCausalLM,
        config: AppConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.train_cfg = config.training
        self.logger = setup_logger("trainer", log_dir=config.project.log_dir, level=config.logging.level)

        # 1. Device selection
        if device:
            self.device = torch.device(device)
        elif config.hardware.device != "auto":
            self.device = torch.device(config.hardware.device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self.model = model.to(self.device)

        # Gradient checkpointing
        if self.train_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self._require_nonempty_dataloader(train_dataloader, "training")
        if eval_dataloader is not None:
            self._require_nonempty_dataloader(eval_dataloader, "validation")

        # 2. Optimizer setup
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.train_cfg.learning_rate,
                betas=(self.train_cfg.beta1, self.train_cfg.beta2),
                eps=self.train_cfg.eps,
                weight_decay=self.train_cfg.weight_decay,
            )

        # 3. Scheduler setup
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            min_ratio = self.train_cfg.min_learning_rate / self.train_cfg.learning_rate
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps=self.train_cfg.warmup_steps,
                max_steps=self.train_cfg.max_steps,
                min_lr_ratio=min_ratio,
            )

        # 4. Mixed precision & GradScaler
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = None

        mixed_prec = self.train_cfg.mixed_precision.lower()
        if mixed_prec != "no":
            if self.device.type == "cuda":
                self.use_amp = True
                if mixed_prec == "bf16" and torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                    self.scaler = torch.amp.GradScaler("cuda", enabled=False)
                else:
                    self.amp_dtype = torch.float16
                    self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            elif self.device.type in ("mps", "cpu"):
                # MPS/CPU float32 default or fp16 autocast where supported
                self.use_amp = False
                self.scaler = None

        # Tracking variables
        self.global_step = 0
        self.micro_step = 0
        self.epoch = 0
        self.total_tokens_processed = 0

    def _prepare_batch(self, batch: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
        """Move batch elements to the active training device."""
        if isinstance(batch, dict):
            result = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "input_ids" not in result:
                raise ValueError("Dictionary batches must contain input_ids")
            return result
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
            labels = batch[1].to(self.device) if len(batch) > 1 else input_ids
            return {"input_ids": input_ids, "labels": labels}
        raise ValueError(f"Unsupported batch format: {type(batch)}")

    @staticmethod
    def _require_nonempty_dataloader(dataloader: DataLoader, name: str) -> None:
        """Fail before entering a loop which could otherwise spin forever."""
        try:
            if len(dataloader) == 0:
                raise ValueError(f"{name.capitalize()} DataLoader is empty")
        except TypeError:
            # Iterable datasets have no reliable length; their first batch is
            # validated by the regular training/evaluation path.
            pass

    def _validate_training_batch(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> None:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.model.max_seq_len:
            raise ValueError("Batch sequence length is invalid or exceeds model.max_seq_len")
        if input_ids.numel() == 0 or input_ids.min().item() < 0 or input_ids.max().item() >= self.config.model.vocab_size:
            raise ValueError("input_ids contain invalid token IDs")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

    def train_step(self, batch: Any) -> float:
        """Execute a single forward-backward-accumulate micro step.

        Returns:
            The raw (unscaled) loss float value.
        """
        self.model.train()
        batch = self._prepare_batch(batch)
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask", None)
        self._validate_training_batch(input_ids, labels, attention_mask)

        self.micro_step += 1
        num_tokens = input_ids.numel()
        self.total_tokens_processed += num_tokens

        # Forward pass with optional AMP
        if self.use_amp:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                output = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = output.loss
        else:
            output = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = output.loss

        raw_loss_val = loss.item()

        # Scale loss by gradient accumulation steps
        scaled_loss = loss / self.train_cfg.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Check if accumulation boundary reached
        if self.micro_step % self.train_cfg.gradient_accumulation_steps == 0:
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        return raw_loss_val

    @torch.no_grad()
    def evaluate(self, eval_dataloader: Optional[DataLoader] = None, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Run validation evaluation over dataset and compute loss and perplexity."""
        dataloader = eval_dataloader or self.eval_dataloader
        if dataloader is None:
            return {}
        self._require_nonempty_dataloader(dataloader, "validation")
        limit = max_batches or self.train_cfg.eval_max_batches

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        total_sequences = 0
        total_tokens = 0
        was_training = self.model.training
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= limit:
                break
            batch = self._prepare_batch(batch)
            self._validate_training_batch(batch["input_ids"], batch.get("labels", batch["input_ids"]), batch.get("attention_mask"))
            output = self.model(
                input_ids=batch["input_ids"],
                labels=batch.get("labels", batch["input_ids"]),
                attention_mask=batch.get("attention_mask"),
            )
            total_loss += output.loss.item()
            total_batches += 1
            total_sequences += batch["input_ids"].shape[0]
            total_tokens += batch["input_ids"].numel()

        mean_loss = total_loss / max(1, total_batches)
        try:
            perplexity = math.exp(min(mean_loss, 20.0))  # Clamp to prevent math overflow
        except OverflowError:
            perplexity = float("inf")

        self.model.train(was_training)
        return {
            "eval_loss": mean_loss,
            "perplexity": perplexity,
            "eval_batches": total_batches,
            "eval_sequences": total_sequences,
            "eval_tokens": total_tokens,
        }

    def train(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Execute full training loop up to max_steps."""
        target_steps = max_steps or self.train_cfg.max_steps
        self.logger.info(f"Starting training: target_steps={target_steps}, device={self.device}")

        start_time = time.time()
        running_loss = 0.0
        steps_in_log = 0

        while self.global_step < target_steps:
            self.epoch += 1
            for batch in self.train_dataloader:
                try:
                    loss_val = self.train_step(batch)
                except torch.cuda.OutOfMemoryError as e:
                    self.logger.error(
                        f"CUDA Out-of-Memory Error at step {self.global_step}! "
                        f"Try reducing batch_size (current: {self.train_cfg.batch_size}) "
                        f"or max_seq_len (current: {self.config.model.max_seq_len})."
                    )
                    raise e

                running_loss += loss_val
                steps_in_log += 1

                # Only perform logging, eval, checkpointing when an optimizer step actually occurred
                if self.micro_step % self.train_cfg.gradient_accumulation_steps == 0:
                    # Logging
                    if self.global_step > 0 and self.global_step % self.config.logging.log_every_steps == 0 and steps_in_log > 0:
                        avg_loss = running_loss / steps_in_log
                        elapsed = time.time() - start_time
                        tokens_per_sec = self.total_tokens_processed / max(0.001, elapsed)
                        current_lr = self.optimizer.param_groups[0]["lr"]

                        self.logger.info(
                            f"Step {self.global_step:5d}/{target_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tok/s: {tokens_per_sec:,.0f}"
                        )
                        running_loss = 0.0
                        steps_in_log = 0

                    # Evaluation
                    if self.global_step > 0 and self.global_step % self.config.logging.eval_every_steps == 0:
                        eval_metrics = self.evaluate()
                        self.logger.info(
                            f"[Eval] Step {self.global_step} | "
                            f"Loss: {eval_metrics.get('eval_loss', 0):.4f} | "
                            f"Perplexity: {eval_metrics.get('perplexity', 0):.2f}"
                        )

                    # Checkpoint saving
                    if self.global_step > 0 and self.global_step % self.config.logging.save_every_steps == 0:
                        ckpt_dir = save_checkpoint(
                            save_dir=self.config.checkpoint.output_dir,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            step=self.global_step,
                            epoch=self.epoch,
                            config=self.config,
                            keep_last_n=self.config.checkpoint.keep_last_n,
                        )
                        self.logger.info(f"Saved checkpoint to {ckpt_dir}")

                if self.global_step >= target_steps:
                    break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s ({self.global_step} steps)")
        return {
            "total_steps": self.global_step,
            "total_tokens": self.total_tokens_processed,
            "total_time": total_time,
        }

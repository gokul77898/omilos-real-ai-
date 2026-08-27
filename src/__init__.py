"""Indian Legal Reasoning Model (Omilos Own AI) - Phase 5 Foundation."""

__version__ = "0.5.0"

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.attention import GroupedQueryAttention, repeat_kv
    from src.block import TransformerBlock, count_parameters
    from src.checkpoint import load_checkpoint, save_checkpoint
    from src.config import (
        AppConfig,
        CheckpointConfig,
        ConfigValidationError,
        HardwareConfig,
        LoggingConfig,
        ModelConfig,
        ProjectConfig,
        TokenizerConfig,
        TokenizerSpecialTokens,
        TrainingConfig,
        ensure_runtime_dirs,
        load_config,
    )
    from src.hardware import (
        GPUInfo,
        HardwareInfo,
        format_hardware_report,
        get_hardware_info,
        run_pytorch_sanity_check,
    )
    from src.logging_utils import setup_logger
    from src.loss import compute_causal_lm_loss
    from src.mlp import SwiGLUMLP
    from src.model import CausalLMOutput, LegalCausalLM
    from src.norm import RMSNorm
    from src.rope import RotaryEmbedding, apply_rotary_emb, rotate_half
    from src.scheduler import get_cosine_schedule_with_warmup
    from src.seed import set_seed
    from src.tokenizer import LegalTokenizer, TokenizerOutput
    from src.trainer import Trainer

_EXPORTS = {
    # Config
    "AppConfig": "src.config",
    "CheckpointConfig": "src.config",
    "ConfigValidationError": "src.config",
    "HardwareConfig": "src.config",
    "LoggingConfig": "src.config",
    "ModelConfig": "src.config",
    "ProjectConfig": "src.config",
    "TokenizerConfig": "src.config",
    "TokenizerSpecialTokens": "src.config",
    "TrainingConfig": "src.config",
    "ensure_runtime_dirs": "src.config",
    "load_config": "src.config",
    # Hardware & Logging
    "GPUInfo": "src.hardware",
    "HardwareInfo": "src.hardware",
    "format_hardware_report": "src.hardware",
    "get_hardware_info": "src.hardware",
    "run_pytorch_sanity_check": "src.hardware",
    "setup_logger": "src.logging_utils",
    "set_seed": "src.seed",
    # Tokenizer
    "LegalTokenizer": "src.tokenizer",
    "TokenizerOutput": "src.tokenizer",
    # Architecture Components (Phase 3)
    "RMSNorm": "src.norm",
    "RotaryEmbedding": "src.rope",
    "rotate_half": "src.rope",
    "apply_rotary_emb": "src.rope",
    "GroupedQueryAttention": "src.attention",
    "repeat_kv": "src.attention",
    "SwiGLUMLP": "src.mlp",
    "TransformerBlock": "src.block",
    "count_parameters": "src.block",
    # Full Language Model (Phase 4)
    "LegalCausalLM": "src.model",
    "CausalLMOutput": "src.model",
    "compute_causal_lm_loss": "src.loss",
    # Training Engine (Phase 5)
    "Trainer": "src.trainer",
    "get_cosine_schedule_with_warmup": "src.scheduler",
    "save_checkpoint": "src.checkpoint",
    "load_checkpoint": "src.checkpoint",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = __import__(_EXPORTS[name], fromlist=[name])
        return getattr(module, name)
    raise AttributeError(f"module 'src' has no attribute '{name}'")


__all__ = list(_EXPORTS.keys()) + ["__version__"]

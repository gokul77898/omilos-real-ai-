"""Configuration management and schema validation for the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
import yaml


class ConfigValidationError(Exception):
    """Raised when configuration file parsing or validation fails."""
    pass


@dataclass
class ProjectConfig:
    """Project metadata and directory settings."""
    name: str = "indian-legal-reasoning"
    seed: int = 42
    output_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class HardwareConfig:
    """Hardware device and precision configuration."""
    device: str = "auto"
    mixed_precision: str = "auto"


@dataclass
class TokenizerSpecialTokens:
    """Explicit special tokens for causal language modeling."""
    pad: str = "<pad>"
    unk: str = "<unk>"
    bos: str = "<s>"
    eos: str = "</s>"

    def to_list(self) -> List[str]:
        return [self.pad, self.unk, self.bos, self.eos]


@dataclass
class TokenizerConfig:
    """Tokenizer hyperparameter and model settings."""
    vocab_size: int = 32000
    model_type: str = "bpe"
    min_frequency: int = 2
    special_tokens: TokenizerSpecialTokens = field(default_factory=TokenizerSpecialTokens)


@dataclass
class ModelConfig:
    """Transformer model architecture hyperparameters."""
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    # Linear RoPE position interpolation.  A factor of 1.0 is plain RoPE.
    rope_scaling_type: str = "linear"
    rope_scaling_factor: float = 1.0
    # Bounded causal local-attention controls.  They do not change parameter shapes.
    attention_window: int = 2048
    attention_chunk_size: int = 1024
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class TrainingConfig:
    """Optimization and training loop hyperparameters."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 500
    max_steps: int = 10000
    max_grad_norm: float = 1.0
    mixed_precision: str = "auto"
    gradient_checkpointing: bool = False
    eval_max_batches: int = 50


@dataclass
class LoggingConfig:
    """Logging verbosity and destination settings."""
    level: str = "INFO"
    log_every_steps: int = 10
    eval_every_steps: int = 100
    save_every_steps: int = 500


@dataclass
class CheckpointConfig:
    """Model checkpoint persistence settings."""
    output_dir: str = "checkpoints"
    keep_last_n: int = 3


@dataclass
class AppConfig:
    """Root configuration object uniting all sub-configurations."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def _validate_dict_structure(raw: dict[str, Any]) -> None:
    """Validate that required sections and fields exist in the raw YAML dictionary."""
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Expected YAML root to be a dictionary/mapping, got {type(raw).__name__}")

    required_sections = ["project", "hardware", "model", "training", "logging"]
    for section in required_sections:
        if section not in raw:
            raise ConfigValidationError(f"Missing required configuration section: '{section}'")
        if not isinstance(raw[section], dict):
            raise ConfigValidationError(f"Configuration section '{section}' must be a dictionary")

    # Project validation
    p = raw["project"]
    if not p.get("name"):
        raise ConfigValidationError("Field 'project.name' must be a non-empty string.")
    if not isinstance(p.get("seed", 0), int):
        raise ConfigValidationError("Field 'project.seed' must be an integer.")

    # Tokenizer validation (optional section or validated if present)
    if "tokenizer" in raw:
        t_raw = raw["tokenizer"]
        if not isinstance(t_raw, dict):
            raise ConfigValidationError("Configuration section 'tokenizer' must be a dictionary")
        if not isinstance(t_raw.get("vocab_size"), int) or t_raw["vocab_size"] < 256:
            raise ConfigValidationError("Field 'tokenizer.vocab_size' must be an integer >= 256")
        if not isinstance(t_raw.get("min_frequency", 0), int) or t_raw.get("min_frequency", 0) < 0:
            raise ConfigValidationError("Field 'tokenizer.min_frequency' must be a non-negative integer")

    # Model validation
    m = raw["model"]
    for field_name in ["vocab_size", "hidden_size", "num_layers", "num_attention_heads", "num_kv_heads", "max_seq_len"]:
        val = m.get(field_name)
        if val is None or not isinstance(val, int) or val <= 0:
            raise ConfigValidationError(f"Field 'model.{field_name}' must be a positive integer, got {val}")

    if m["hidden_size"] % m["num_attention_heads"] != 0:
        raise ConfigValidationError(
            f"model.hidden_size ({m['hidden_size']}) must be divisible by model.num_attention_heads ({m['num_attention_heads']})"
        )
    if m["num_attention_heads"] % m["num_kv_heads"] != 0:
        raise ConfigValidationError(
            f"model.num_attention_heads ({m['num_attention_heads']}) must be divisible by model.num_kv_heads ({m['num_kv_heads']})"
        )

    if not isinstance(m.get("intermediate_size", 1), int) or m.get("intermediate_size", 1) <= 0:
        raise ConfigValidationError("Field 'model.intermediate_size' must be a positive integer.")
    if not isinstance(m.get("rope_theta", 1.0), (int, float)) or m.get("rope_theta", 1.0) <= 0:
        raise ConfigValidationError("Field 'model.rope_theta' must be a positive number.")
    if not isinstance(m.get("rms_norm_eps", 1e-6), (int, float)) or m.get("rms_norm_eps", 1e-6) <= 0:
        raise ConfigValidationError("Field 'model.rms_norm_eps' must be a positive float.")
    if not isinstance(m.get("tie_word_embeddings", True), bool):
        raise ConfigValidationError("Field 'model.tie_word_embeddings' must be a boolean.")
    if m.get("rope_scaling_type", "linear") not in {"linear", "none"}:
        raise ConfigValidationError("Field 'model.rope_scaling_type' must be 'linear' or 'none'.")
    if not isinstance(m.get("rope_scaling_factor", 1.0), (int, float)) or m.get("rope_scaling_factor", 1.0) < 1.0:
        raise ConfigValidationError("Field 'model.rope_scaling_factor' must be a number >= 1.")
    for name in ("attention_window", "attention_chunk_size"):
        if not isinstance(m.get(name, m["max_seq_len"]), int) or m.get(name, m["max_seq_len"]) <= 0:
            raise ConfigValidationError(f"Field 'model.{name}' must be a positive integer.")

    # Training validation
    t = raw["training"]
    if not isinstance(t.get("batch_size"), int) or t["batch_size"] <= 0:
        raise ConfigValidationError("Field 'training.batch_size' must be a positive integer.")
    if not isinstance(t.get("gradient_accumulation_steps"), int) or t["gradient_accumulation_steps"] <= 0:
        raise ConfigValidationError("Field 'training.gradient_accumulation_steps' must be a positive integer.")
    if not isinstance(t.get("learning_rate"), (int, float)) or t["learning_rate"] <= 0:
        raise ConfigValidationError("Field 'training.learning_rate' must be a positive number.")
    if not isinstance(t.get("weight_decay"), (int, float)) or t["weight_decay"] < 0:
        raise ConfigValidationError("Field 'training.weight_decay' must be a non-negative number.")
    if not isinstance(t.get("max_steps"), int) or t["max_steps"] <= 0:
        raise ConfigValidationError("Field 'training.max_steps' must be a positive integer.")
    if not isinstance(t.get("warmup_steps"), int) or t["warmup_steps"] < 0:
        raise ConfigValidationError("Field 'training.warmup_steps' must be a non-negative integer.")
    if not isinstance(t.get("eval_max_batches", 50), int) or t.get("eval_max_batches", 50) <= 0:
        raise ConfigValidationError("Field 'training.eval_max_batches' must be a positive integer.")

    # Logging validation
    log_level = str(raw["logging"].get("level", "")).upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        raise ConfigValidationError(f"Field 'logging.level' must be one of {valid_levels}, got '{log_level}'")


def load_config(config_path: str | Path = "configs/base.yaml") -> AppConfig:
    """Load, parse, and validate YAML configuration into an AppConfig instance."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigValidationError(f"Configuration file not found: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Error parsing YAML file '{path}': {exc}") from exc
    except Exception as exc:
        raise ConfigValidationError(f"Error reading configuration file '{path}': {exc}") from exc

    if raw_data is None:
        raise ConfigValidationError(f"Configuration file '{path}' is empty.")

    _validate_dict_structure(raw_data)

    project_cfg = ProjectConfig(**raw_data["project"])
    hardware_cfg = HardwareConfig(**raw_data["hardware"])

    if "tokenizer" in raw_data:
        tok_data = raw_data["tokenizer"]
        specials_data = tok_data.get("special_tokens", {})
        specials_cfg = TokenizerSpecialTokens(
            pad=specials_data.get("pad", "<pad>"),
            unk=specials_data.get("unk", "<unk>"),
            bos=specials_data.get("bos", "<s>"),
            eos=specials_data.get("eos", "</s>"),
        )
        tokenizer_cfg = TokenizerConfig(
            vocab_size=int(tok_data.get("vocab_size", 32000)),
            model_type=str(tok_data.get("model_type", "bpe")),
            min_frequency=int(tok_data.get("min_frequency", 2)),
            special_tokens=specials_cfg,
        )
    else:
        tokenizer_cfg = TokenizerConfig()

    m_data = raw_data["model"]
    model_cfg = ModelConfig(
        vocab_size=int(m_data["vocab_size"]),
        hidden_size=int(m_data["hidden_size"]),
        num_layers=int(m_data["num_layers"]),
        num_attention_heads=int(m_data["num_attention_heads"]),
        num_kv_heads=int(m_data["num_kv_heads"]),
        intermediate_size=int(m_data.get("intermediate_size", int(m_data["hidden_size"] * 8 / 3))),
        max_seq_len=int(m_data["max_seq_len"]),
        rope_theta=float(m_data.get("rope_theta", 10000.0)),
        rope_scaling_type=str(m_data.get("rope_scaling_type", "linear")),
        rope_scaling_factor=float(m_data.get("rope_scaling_factor", 1.0)),
        attention_window=int(m_data.get("attention_window", m_data["max_seq_len"])),
        attention_chunk_size=int(m_data.get("attention_chunk_size", min(1024, m_data["max_seq_len"]))),
        rms_norm_eps=float(m_data.get("rms_norm_eps", 1e-6)),
        tie_word_embeddings=bool(m_data.get("tie_word_embeddings", True)),
    )

    t_data = raw_data["training"]
    training_cfg = TrainingConfig(
        batch_size=int(t_data["batch_size"]),
        gradient_accumulation_steps=int(t_data["gradient_accumulation_steps"]),
        learning_rate=float(t_data["learning_rate"]),
        min_learning_rate=float(t_data.get("min_learning_rate", t_data["learning_rate"] * 0.1)),
        weight_decay=float(t_data["weight_decay"]),
        beta1=float(t_data.get("beta1", 0.9)),
        beta2=float(t_data.get("beta2", 0.95)),
        eps=float(t_data.get("eps", 1e-8)),
        warmup_steps=int(t_data["warmup_steps"]),
        max_steps=int(t_data["max_steps"]),
        max_grad_norm=float(t_data.get("max_grad_norm", 1.0)),
        mixed_precision=str(t_data.get("mixed_precision", "auto")),
        gradient_checkpointing=bool(t_data.get("gradient_checkpointing", False)),
        eval_max_batches=int(t_data.get("eval_max_batches", 50)),
    )

    l_data = raw_data["logging"]
    logging_cfg = LoggingConfig(
        level=str(l_data.get("level", "INFO")).upper(),
        log_every_steps=int(l_data.get("log_every_steps", 10)),
        eval_every_steps=int(l_data.get("eval_every_steps", 100)),
        save_every_steps=int(l_data.get("save_every_steps", 500)),
    )

    ckpt_data = raw_data.get("checkpoint", {})
    checkpoint_cfg = CheckpointConfig(
        output_dir=str(ckpt_data.get("output_dir", "checkpoints")),
        keep_last_n=int(ckpt_data.get("keep_last_n", 3)),
    )

    return AppConfig(
        project=project_cfg,
        hardware=hardware_cfg,
        tokenizer=tokenizer_cfg,
        model=model_cfg,
        training=training_cfg,
        logging=logging_cfg,
        checkpoint=checkpoint_cfg,
    )


def ensure_runtime_dirs(config: AppConfig, base_dir: str | Path | None = None) -> dict[str, Path]:
    """Explicitly create runtime directories defined in the configuration."""
    root = Path(base_dir) if base_dir else Path.cwd()
    created_dirs = {
        "output_dir": root / config.project.output_dir,
        "checkpoint_dir": root / config.project.checkpoint_dir,
        "log_dir": root / config.project.log_dir,
    }
    for dir_path in created_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return created_dirs

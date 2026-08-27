"""Tests for configuration loading, validation, and directory management."""

from pathlib import Path
import tempfile
import pytest
import yaml

from src.config import (
    AppConfig,
    ConfigValidationError,
    ensure_runtime_dirs,
    load_config,
)


def test_load_base_config_success():
    """Verify loading default base.yaml succeeds with correct types."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config = load_config(config_path)
    assert isinstance(config, AppConfig)
    assert config.project.name == "indian-legal-reasoning"
    assert config.project.seed == 42
    assert config.model.vocab_size == 32000
    assert config.model.hidden_size == 512
    assert config.model.num_layers == 8
    assert config.model.num_attention_heads == 8
    assert config.model.num_kv_heads == 4
    assert config.model.intermediate_size == 1408
    assert config.training.batch_size == 4
    assert config.training.learning_rate == 0.0003
    assert config.logging.level == "INFO"


def test_missing_file_raises_error():
    """Verify attempting to load a non-existent configuration raises ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="Configuration file not found"):
        load_config("configs/non_existent_config.yaml")


def test_missing_required_section_raises_error():
    """Verify configuration missing a top-level section fails validation."""
    incomplete_data = {
        "project": {"name": "test", "seed": 42},
        "hardware": {"device": "cpu"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(incomplete_data, f)
        temp_path = f.name

    try:
        with pytest.raises(ConfigValidationError, match="Missing required configuration section"):
            load_config(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_invalid_model_parameters_raises_error():
    """Verify invalid model hyperparameter values (e.g. negative layers) fail validation."""
    invalid_data = {
        "project": {"name": "test", "seed": 42},
        "hardware": {"device": "cpu", "mixed_precision": "no"},
        "model": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": -5,
            "num_attention_heads": 12,
            "num_kv_heads": 4,
            "max_seq_len": 2048,
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_steps": 100,
            "warmup_steps": 10,
        },
        "logging": {"level": "INFO"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(invalid_data, f)
        temp_path = f.name

    try:
        with pytest.raises(ConfigValidationError, match="must be a positive integer"):
            load_config(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_invalid_logging_level_raises_error():
    """Verify invalid logging levels fail validation."""
    invalid_data = {
        "project": {"name": "test", "seed": 42},
        "hardware": {"device": "cpu", "mixed_precision": "no"},
        "model": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "num_kv_heads": 4,
            "max_seq_len": 2048,
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_steps": 100,
            "warmup_steps": 10,
        },
        "logging": {"level": "INVALID_LEVEL_NAME"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(invalid_data, f)
        temp_path = f.name

    try:
        with pytest.raises(ConfigValidationError, match="Field 'logging.level' must be one of"):
            load_config(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_ensure_runtime_dirs_creates_directories():
    """Verify ensure_runtime_dirs explicitly creates checkpoints, logs, runs directories."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config(config_path)
        dirs = ensure_runtime_dirs(config, base_dir=tmpdir)
        for d in dirs.values():
            assert d.exists()
            assert d.is_dir()

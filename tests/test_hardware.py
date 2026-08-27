"""Tests for hardware detection, reporting, and PyTorch tensor sanity checks."""

import torch

from src.hardware import (
    HardwareInfo,
    format_hardware_report,
    get_hardware_info,
    run_pytorch_sanity_check,
)


def test_get_hardware_info_structure():
    """Verify get_hardware_info returns complete structured data."""
    info = get_hardware_info()
    assert isinstance(info, HardwareInfo)
    assert isinstance(info.pytorch_version, str)
    assert isinstance(info.cuda_available, bool)
    assert isinstance(info.gpu_count, int)
    assert isinstance(info.cpu_cores_logical, int)
    assert info.cpu_cores_logical > 0
    assert info.total_ram_gb > 0
    assert info.recommended_device in {"cuda", "mps", "cpu"}
    assert info.recommended_precision in {"bf16", "fp16", "no"}


def test_format_hardware_report_contains_key_headers():
    """Verify format_hardware_report outputs standard section headers."""
    report = format_hardware_report()
    assert "HARDWARE REPORT" in report
    assert "PyTorch:" in report
    assert "CPU:" in report
    assert "RAM:" in report
    assert "Recommended device:" in report
    assert "Recommended precision:" in report


def test_run_pytorch_sanity_check_cpu():
    """Verify PyTorch tensor computation and backward pass pass on CPU."""
    res = run_pytorch_sanity_check(device="cpu")
    assert res["success"] is True
    assert res["device"] == "cpu"
    assert res["x_grad_shape"] == [4, 4]
    assert res["w_grad_shape"] == [4, 4]
    assert res["b_grad_shape"] == [4]


def test_run_pytorch_sanity_check_auto_device():
    """Verify PyTorch tensor sanity check works on auto-detected device."""
    res = run_pytorch_sanity_check()
    assert res["success"] is True
    assert res["loss_value"] is not None

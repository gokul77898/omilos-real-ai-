"""Hardware detection, environment diagnostics, and accelerator reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import platform
import subprocess
from typing import Any, List, Optional
import psutil
import torch


@dataclass
class GPUInfo:
    """Detailed hardware metadata for a single GPU device."""
    index: int
    name: str
    total_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float
    bf16_supported: bool
    fp16_supported: bool


@dataclass
class HardwareInfo:
    """Comprehensive system hardware and PyTorch backend status."""
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpus: List[GPUInfo]
    mps_available: bool
    mps_built: bool
    cpu_model: str
    cpu_cores_logical: int
    cpu_cores_physical: int
    total_ram_gb: float
    available_ram_gb: float
    recommended_device: str
    recommended_precision: str


def _get_cpu_model_name() -> str:
    """Retrieve detailed CPU brand/model across macOS, Linux, and Windows."""
    system = platform.system()
    try:
        if system == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            model = res.stdout.strip()
            if model:
                return model
        elif system == "Linux":
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":", 1)[1].strip()
        elif system == "Windows":
            return platform.processor() or "Unknown CPU"
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def _check_bf16_support() -> bool:
    """Detect if current CUDA device supports native bfloat16 computation."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        try:
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
        except Exception:
            return False


def _check_fp16_support() -> bool:
    """Detect if current device supports FP16 computation."""
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            return (major, minor) >= (5, 3)
        except Exception:
            return True
    return False


def get_hardware_info() -> HardwareInfo:
    """Detect and collect system hardware, GPU, and PyTorch runtime information.

    Returns:
        HardwareInfo dataclass containing structured hardware parameters.
    """
    pytorch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    gpus: List[GPUInfo] = []
    if cuda_available and gpu_count > 0:
        for idx in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(idx)
                name = props.name
                total_mem = props.total_memory / (1024 ** 3)
                allocated_mem = torch.cuda.memory_allocated(idx) / (1024 ** 3)
                reserved_mem = torch.cuda.memory_reserved(idx) / (1024 ** 3)
                bf16 = _check_bf16_support()
                fp16 = _check_fp16_support()
            except Exception:
                name = f"GPU {idx}"
                total_mem = 0.0
                allocated_mem = 0.0
                reserved_mem = 0.0
                bf16 = False
                fp16 = False

            gpus.append(
                GPUInfo(
                    index=idx,
                    name=name,
                    total_memory_gb=round(total_mem, 2),
                    allocated_memory_gb=round(allocated_mem, 3),
                    reserved_memory_gb=round(reserved_mem, 3),
                    bf16_supported=bf16,
                    fp16_supported=fp16,
                )
            )

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()

    # CPU & RAM diagnostics
    cpu_model = _get_cpu_model_name()
    cpu_cores_logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_cores_logical

    vmem = psutil.virtual_memory()
    total_ram = round(vmem.total / (1024 ** 3), 2)
    available_ram = round(vmem.available / (1024 ** 3), 2)

    # Recommendations
    if cuda_available and gpu_count > 0:
        recommended_device = "cuda"
        if gpus[0].bf16_supported:
            recommended_precision = "bf16"
        elif gpus[0].fp16_supported:
            recommended_precision = "fp16"
        else:
            recommended_precision = "no"
    elif mps_available:
        recommended_device = "mps"
        recommended_precision = "fp16"
    else:
        recommended_device = "cpu"
        recommended_precision = "no"

    return HardwareInfo(
        pytorch_version=pytorch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_count=gpu_count,
        gpus=gpus,
        mps_available=mps_available,
        mps_built=mps_built,
        cpu_model=cpu_model,
        cpu_cores_logical=cpu_cores_logical,
        cpu_cores_physical=cpu_cores_physical,
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        recommended_device=recommended_device,
        recommended_precision=recommended_precision,
    )


def format_hardware_report(info: Optional[HardwareInfo] = None) -> str:
    """Format the hardware information into a clean, human-readable report.

    Args:
        info: Optional HardwareInfo instance. If omitted, detects hardware automatically.

    Returns:
        Formatted multi-line report string.
    """
    if info is None:
        info = get_hardware_info()

    lines = [
        "=" * 50,
        "HARDWARE REPORT",
        "=" * 50,
        "",
        f"PyTorch: {info.pytorch_version}",
        f"CUDA: {'Available (' + str(info.cuda_version) + ')' if info.cuda_available else 'Not available'}",
        f"GPU count: {info.gpu_count}",
    ]

    if info.gpu_count > 0:
        for gpu in info.gpus:
            lines.extend([
                "",
                f"GPU {gpu.index}:",
                f"Name: {gpu.name}",
                f"VRAM: {gpu.total_memory_gb:.2f} GB",
                f"BF16: {'Supported' if gpu.bf16_supported else 'Not supported'}",
                f"FP16: {'Supported' if gpu.fp16_supported else 'Not supported'}",
            ])
    else:
        if info.mps_available:
            lines.extend([
                "",
                "Apple Silicon MPS: Available",
            ])
        lines.extend([
            "",
            "No CUDA GPUs detected (Running in CPU/MPS mode)",
        ])

    lines.extend([
        "",
        f"CPU: {info.cpu_model}",
        f"CPU cores: {info.cpu_cores_logical} (Physical: {info.cpu_cores_physical})",
        f"RAM: {info.total_ram_gb:.2f} GB (Available: {info.available_ram_gb:.2f} GB)",
        "",
        f"Recommended device: {info.recommended_device}",
        f"Recommended precision: {info.recommended_precision}",
        "=" * 50,
    ])

    return "\n".join(lines)


def run_pytorch_sanity_check(device: Optional[str] = None) -> dict[str, Any]:
    """Perform a minimal PyTorch sanity check to verify tensor operations and backward pass.

    Steps:
    1. Creates small tensors with gradient tracking.
    2. Moves tensors to target device (CUDA, MPS, or CPU).
    3. Performs matrix multiplication and addition.
    4. Runs backward pass.
    5. Verifies gradients exist and are finite.

    Args:
        device: Target device ('cpu', 'cuda', 'mps', or auto-detected if None).

    Returns:
        Dictionary with test execution status and metadata.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            target_device = torch.device("mps")
        else:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    # 1. Create tiny tensors with requires_grad
    x = torch.randn(4, 4, requires_grad=True, device=target_device)
    w = torch.randn(4, 4, requires_grad=True, device=target_device)
    b = torch.randn(4, requires_grad=True, device=target_device)

    # 2. Matrix multiplication and addition
    y = torch.matmul(x, w) + b
    loss = y.sum()

    # 3. Backward propagation
    loss.backward()

    # 4. Verification
    has_x_grad = x.grad is not None and torch.isfinite(x.grad).all().item()
    has_w_grad = w.grad is not None and torch.isfinite(w.grad).all().item()
    has_b_grad = b.grad is not None and torch.isfinite(b.grad).all().item()
    success = bool(has_x_grad and has_w_grad and has_b_grad)

    return {
        "success": success,
        "device": str(target_device),
        "loss_value": float(loss.item()),
        "x_grad_shape": list(x.grad.shape) if x.grad is not None else None,
        "w_grad_shape": list(w.grad.shape) if w.grad is not None else None,
        "b_grad_shape": list(b.grad.shape) if b.grad is not None else None,
    }

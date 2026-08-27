#!/usr/bin/env python3
"""Hardware diagnostic CLI script for Omilos Own AI / Indian Legal Reasoning."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware import format_hardware_report, get_hardware_info, run_pytorch_sanity_check


def main() -> None:
    """Run hardware diagnostics and PyTorch tensor verification."""
    info = get_hardware_info()
    report = format_hardware_report(info)
    print(report)

    print("\nRunning PyTorch tensor sanity check...")
    sanity = run_pytorch_sanity_check(device=info.recommended_device)
    if sanity["success"]:
        print(f"✓ PyTorch compute sanity check PASSED on device '{sanity['device']}' (loss: {sanity['loss_value']:.4f})")
    else:
        print(f"✗ PyTorch compute sanity check FAILED on device '{sanity['device']}'")
        sys.exit(1)


if __name__ == "__main__":
    main()

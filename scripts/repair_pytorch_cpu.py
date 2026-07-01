"""Repair and verify a CPU PyTorch install for the AML environment.

This script first tries to import torch. If import succeeds, it does not modify
anything. If import fails, it reinstalls CPU PyTorch wheels using the official
PyTorch CPU wheel index and verifies the install with a small tensor operation.
"""

from __future__ import annotations

import subprocess
import sys

INSTALL_COMMAND = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--force-reinstall",
    "torch",
    "torchvision",
    "torchaudio",
    "--index-url",
    "https://download.pytorch.org/whl/cpu",
]


def verify_torch() -> bool:
    """Return True when torch imports and can create a tensor."""

    try:
        import torch

        print(f"torch import ok: {torch.__version__}")
        print(torch.rand(2, 2))
        return True
    except Exception as exc:
        print(f"torch import failed: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    """Verify torch and reinstall CPU wheels only when needed."""

    if verify_torch():
        print("No PyTorch repair needed.")
        return

    print("Reinstalling CPU PyTorch wheels...")
    print(" ".join(INSTALL_COMMAND))
    subprocess.check_call(INSTALL_COMMAND)
    if not verify_torch():
        raise SystemExit("PyTorch repair completed but verification still failed.")
    print("PyTorch repair completed.")


if __name__ == "__main__":
    main()

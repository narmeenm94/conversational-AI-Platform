"""Prepare optional native dependencies before the ML stack is imported."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def prepare_windows_stable_abi() -> None:
    """Load a recovery ``python3.dll`` for embedded-Python virtualenvs.

    A normal python.org installation already supplies and loads this DLL, so
    this is a no-op unless a local recovery copy exists in ``.venv/Scripts``.
    """

    if sys.platform != "win32":
        return

    stable_abi_dll = Path(__file__).parent / ".venv" / "Scripts" / "python3.dll"
    if not stable_abi_dll.is_file():
        return

    try:
        ctypes.WinDLL(str(stable_abi_dll))
    except OSError:
        # If the standard runtime already provided it, native imports will work.
        # Otherwise the affected package will raise the useful diagnostic.
        pass


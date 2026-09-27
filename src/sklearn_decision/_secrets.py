"""Resolve credentials that are given as secret references.

A credential may be the secret itself or a 1Password secret reference,
``op://<vault>/<item>/<field>``. References are resolved with the 1Password
CLI (``op read``) the first time a request needs them. The resolved value is
kept only in this module's process memory, never on an estimator, so it is
not pickled, cloned, logged or shown in a repr.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading

from ._answers import DecisionModelError

__all__ = ["resolve_secret", "clear_secret_cache", "is_secret_reference"]

_RESOLVED: dict[str, str] = {}
_LOCK = threading.Lock()


def is_secret_reference(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith("op://")


def clear_secret_cache() -> None:
    """Forget every resolved secret (they are re-read on next use)."""
    with _LOCK:
        _RESOLVED.clear()


def _op_binary() -> str:
    path = os.environ.get("OP_CLI_PATH") or shutil.which("op")
    if not path:
        raise DecisionModelError(
            "This credential is a 1Password reference (op://...), but the 1Password CLI `op` was not found. "
            "Install it (https://developer.1password.com/docs/cli/get-started/), enable "
            "'Integrate with 1Password CLI' in the 1Password app, or set OP_CLI_PATH.", fatal=True)
    return path


def resolve_secret(value: str | None, *, timeout: float = 120.0) -> str | None:
    """Return ``value`` itself, or the secret an ``op://`` reference points to.

    The timeout leaves room for a 1Password approval prompt. Failures raise a
    fatal :class:`DecisionModelError` whose message names the reference but
    never contains secret material.
    """
    if not is_secret_reference(value):
        return value
    with _LOCK:
        if value in _RESOLVED:
            return _RESOLVED[value]
        try:
            proc = subprocess.run([_op_binary(), "read", "--no-newline", value], capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise DecisionModelError(f"`op read {value}` timed out after {timeout:.0f}s (approval pending?)",
                                     fatal=True) from e
        except OSError as e:
            raise DecisionModelError(f"Could not run the 1Password CLI: {e}", fatal=True) from e
        if proc.returncode != 0 or not proc.stdout:
            detail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
            raise DecisionModelError(f"`op read {value}` failed: {detail[0][:300]}", fatal=True)
        _RESOLVED[value] = proc.stdout
        return proc.stdout

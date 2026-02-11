"""Monkeypatch to fix Python 3.12.7+ type annotation issues in third-party libraries.

This module patches libraries that use TYPE_CHECKING imports without
`from __future__ import annotations`, which causes NameError even in Python 3.12.7.

Affected libraries:
- pyupbit 0.2.30+ (Callable, Response)
- PyJWT 2.10+ (EllipticCurve, RSAPrivateKey, etc.)

Issue: These libraries import types inside `if TYPE_CHECKING:` block,
but use them in runtime type annotations, causing NameError.

Solution: Directly patch the module files to add the missing imports.

This patch is automatically applied by importing this module.
"""

from __future__ import annotations

from pathlib import Path


def _patch_pyupbit_errors() -> None:
    """Patch pyupbit.errors.py to fix Callable forward reference.

    Adds `from collections.abc import Callable` outside the TYPE_CHECKING block.
    """
    try:
        # Find pyupbit installation without importing it
        import sysconfig

        site_packages = sysconfig.get_path("purelib")
        if not site_packages:
            return

        errors_file = Path(site_packages) / "pyupbit" / "errors.py"

        if not errors_file.exists():
            return

        # Read the current content
        content = errors_file.read_text(encoding="utf-8")

        # Check if already patched
        if "# PATCHED BY crypto-lab" in content:
            return

        # Add the fix: import Callable at runtime
        patched_content = content.replace(
            "from typing import TYPE_CHECKING, Any\n\nif TYPE_CHECKING:\n    from collections.abc import Callable",
            "# PATCHED BY crypto-lab\n"
            "from typing import TYPE_CHECKING, Any\n"
            "from collections.abc import Callable\n\n"
            "if TYPE_CHECKING:",
        )

        # Also need to import Response at runtime
        patched_content = patched_content.replace(
            "if TYPE_CHECKING:\n\n    from requests import Response",
            "if TYPE_CHECKING:\n    from requests import Response\nelse:\n    Response = object  # Runtime placeholder",
        )

        # Only write if content actually changed
        if patched_content != content:
            errors_file.write_text(patched_content, encoding="utf-8")

    except Exception:
        # Silently fail if patching doesn't work
        pass


def _patch_pyjwt_utils() -> None:
    """Patch jwt/utils.py to fix EllipticCurve forward reference."""
    try:
        import sysconfig

        site_packages = sysconfig.get_path("purelib")
        if not site_packages:
            return

        utils_file = Path(site_packages) / "jwt" / "utils.py"

        if not utils_file.exists():
            return

        content = utils_file.read_text(encoding="utf-8")

        # Check if already patched
        if "# PATCHED BY crypto-lab" in content:
            return

        # Read the file to find the TYPE_CHECKING block
        # We need to move the cryptography imports outside TYPE_CHECKING
        patched_content = content.replace(
            "if TYPE_CHECKING:",
            "# PATCHED BY crypto-lab\n"
            "# Move cryptography imports outside TYPE_CHECKING for Python 3.12+\n"
            "try:\n"
            "    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurve\n"
            "    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey\n"
            "    from cryptography.hazmat.primitives.asymmetric.ed25519 import (\n"
            "        Ed25519PrivateKey,\n"
            "        Ed25519PublicKey,\n"
            "    )\n"
            "except ImportError:\n"
            "    # Fallback if cryptography is not installed\n"
            "    EllipticCurve = object  # type: ignore[misc, assignment]\n"
            "    RSAPrivateKey = object  # type: ignore[misc, assignment]\n"
            "    RSAPublicKey = object  # type: ignore[misc, assignment]\n"
            "    Ed25519PrivateKey = object  # type: ignore[misc, assignment]\n"
            "    Ed25519PublicKey = object  # type: ignore[misc, assignment]\n\n"
            "if TYPE_CHECKING:",
        )

        # Only write if content actually changed
        if patched_content != content:
            utils_file.write_text(patched_content, encoding="utf-8")

    except Exception:
        pass


def _patch_pyjwt_jwks_client() -> None:
    """Patch jwt/jwks_client.py to fix SSLContext forward reference."""
    try:
        import sysconfig

        site_packages = sysconfig.get_path("purelib")
        if not site_packages:
            return

        jwks_file = Path(site_packages) / "jwt" / "jwks_client.py"

        if not jwks_file.exists():
            return

        content = jwks_file.read_text(encoding="utf-8")

        # Check if already patched
        if "# PATCHED BY crypto-lab" in content:
            return

        # Move SSLContext import outside TYPE_CHECKING
        patched_content = content.replace(
            "if TYPE_CHECKING:\n    from ssl import SSLContext",
            "# PATCHED BY crypto-lab\n"
            "from ssl import SSLContext\n\n"
            "if TYPE_CHECKING:\n    pass  # SSLContext moved above",
        )

        # Only write if content actually changed
        if patched_content != content:
            jwks_file.write_text(patched_content, encoding="utf-8")

    except Exception:
        pass


# Apply patches on import
_patch_pyupbit_errors()
_patch_pyjwt_utils()
_patch_pyjwt_jwks_client()

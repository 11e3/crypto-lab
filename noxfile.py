import shutil
from contextlib import suppress
from pathlib import Path

import nox

# Python version to test against
DEFAULT_PYTHON = "3.14"
PYTHON_VERSIONS = ["3.14"]
LINE_LENGTH = "100"


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def format(session: nox.Session) -> None:
    """Auto-format code and sort imports using ruff."""
    session.install(".[dev]")

    # 1. Sort imports (isort equivalent in ruff)
    session.run("ruff", "check", ".", "--select", "I", "--fix")

    # 2. Format code style
    session.run("ruff", "format", ".", f"--line-length={LINE_LENGTH}")

    session.log("✓ Code formatting and import sorting complete")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run linting and code quality checks."""
    session.install(".[dev]")

    # 1. Check all rules including import sorting (I)
    session.run("ruff", "check", ".", "--fix")

    # 2. Verify formatting consistency
    session.run("ruff", "format", ".", "--check", f"--line-length={LINE_LENGTH}")

    session.log("✓ Linting complete")


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run test suite with coverage."""
    session.install(".[test]")
    session.run("pytest", "--cov=src", "-q", "--tb=line")
    session.log("✓ Tests complete")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def type_check(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install(".[dev]")
    session.run("mypy", ".", "--strict", "--no-error-summary")
    session.log("✓ Type checking complete")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build documentation with Sphinx."""
    session.install(".[docs]")

    session.chdir("docs")
    session.run("sphinx-build", "-Q", "-b", "html", "-d", "_build/doctrees", ".", "_build/html")
    session.chdir("..")
    session.log("✓ Documentation built - open docs/_build/html/index.html")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def clean(session: nox.Session) -> None:
    """Clean up generated files and caches."""
    dirs_to_remove = [
        "build",
        "dist",
        "htmlcov",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "docs/_build",
    ]
    files_to_remove = [".coverage", "coverage.xml", "coverage.json"]

    for dir_name in dirs_to_remove:
        with suppress(FileNotFoundError, NotADirectoryError, OSError):
            shutil.rmtree(dir_name)
            session.log(f"  Removed {dir_name}/")

    for file_name in files_to_remove:
        with suppress(FileNotFoundError, OSError):
            Path(file_name).unlink()
            session.log(f"  Removed {file_name}")

    session.log("✓ Cleanup complete")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def pre_commit_check(session: nox.Session) -> None:
    """Run pre-commit checks on all files."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
    session.log("✓ Pre-commit checks complete")

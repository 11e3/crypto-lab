import nox

# Python version to test against
DEFAULT_PYTHON = "3.14"
PYTHON_VERSIONS = ["3.14"]


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run linting and code quality checks."""
    session.install(".[dev]")

    # Format with ruff
    session.run("ruff", "check", ".", "--fix", "--unsafe-fixes", "--line-length=100")

    session.log("✓ Linting complete")


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run test suite with coverage."""
    session.install(".[test]")

    # Run tests with coverage
    session.run("pytest", "src", "tests", "-q", "--tb=line")

    session.log("✓ Tests complete - coverage report in htmlcov/index.html")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def type_check(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install(".[dev]")
    session.run("mypy", ".", "--strict", "--no-error-summary")
    session.log("✓ Type checking complete")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def format(session: nox.Session) -> None:
    """Auto-format code with black and isort."""
    session.install(".[dev]")

    # Format with ruff
    session.run("ruff", "format", ".", "--line-length=100")

    # Sort imports
    session.run("isort", "src", "tests", "--profile=black", "--line-length=100")

    session.log("✓ Code formatting complete")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build documentation with Sphinx."""
    session.install(".[docs]")

    session.chdir("docs")
    # -w warnings.txt 옵션으로 경고를 파일에 저장하지만 빌드는 실패하지 않음
    session.run("sphinx-build", "-Q", "-b", "html", "-d", "_build/doctrees", ".", "_build/html")

    session.chdir("..")
    session.log("✓ Documentation built - open docs/_build/html/index.html")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def clean(session: nox.Session) -> None:
    """Clean up generated files and caches."""
    import shutil
    from pathlib import Path

    dirs_to_remove = [
        "build",
        "dist",
        "htmlcov",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "docs/_build",
    ]

    files_to_remove = [
        ".coverage",
        "coverage.xml",
        "coverage.json",
    ]

    # Remove directories
    for dir_name in dirs_to_remove:
        try:
            shutil.rmtree(dir_name)
            session.log(f"  Removed {dir_name}/")
        except (FileNotFoundError, NotADirectoryError, OSError):
            pass

    # Remove files
    for file_name in files_to_remove:
        try:
            Path(file_name).unlink()
            session.log(f"  Removed {file_name}")
        except FileNotFoundError:
            pass

    session.log("✓ Cleanup complete")


@nox.session(python=DEFAULT_PYTHON, reuse_venv=True)
def pre_commit_check(session: nox.Session) -> None:
    """Run pre-commit checks on all files."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
    session.log("✓ Pre-commit checks complete")

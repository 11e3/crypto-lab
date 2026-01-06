# Makefile for common development tasks
# Uses uv for dependency management

.PHONY: help install install-dev test lint format format-check type-check check fix clean docs docs-clean docs-serve

help:
	@echo "Available commands:"
	@echo "  make install      - Install package in development mode (uv sync)"
	@echo "  make install-dev  - Install with development dependencies (uv sync --extra dev)"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters (ruff check)"
	@echo "  make format       - Format code (ruff format)"
	@echo "  make format-check - Check code formatting"
	@echo "  make type-check   - Run type checker (mypy)"
	@echo "  make check        - Run all checks (lint, format-check, type-check, test)"
	@echo "  make fix           - Format code and fix linting issues"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docs         - Build Sphinx documentation"
	@echo "  make docs-clean   - Clean documentation build artifacts"
	@echo "  make docs-serve   - Build and serve documentation locally"

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

type-check:
	uv run mypy src/

check: lint format-check type-check test
	@echo "All checks passed!"

fix: format
	uv run ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docs:
	@echo "Building Sphinx documentation..."
	cd docs && uv run sphinx-build -b html . _build/html
	@echo "Documentation built in docs/_build/html/"

docs-clean:
	rm -rf docs/_build/
	rm -rf docs/_static/
	rm -rf docs/_templates/
	find docs -type f -name "*.rst" ! -name "index.rst" ! -name "api/index.rst" -delete 2>/dev/null || true

docs-serve: docs
	@echo "Starting documentation server..."
	@echo "Open http://localhost:8000 in your browser"
	cd docs/_build/html && python -m http.server 8000

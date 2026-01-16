# Multi-stage build for production-grade crypto quant trading system
# Python 3.12.7 with monkeypatch support for third-party type annotation issues

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.12.7-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv==0.5.11

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies into .venv
# --frozen: Use exact versions from uv.lock
# --all-extras: Install all optional dependencies (web, dev, etc.)
RUN uv sync --frozen --all-extras

# ============================================================================
# Stage 2: Production - Minimal runtime image
# ============================================================================
FROM python:3.12.7-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    mkdir -p /app /app/data /app/logs && \
    chown -R botuser:botuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=botuser:botuser /app/.venv /app/.venv

# Copy application code
COPY --chown=botuser:botuser src/ /app/src/
COPY --chown=botuser:botuser pyproject.toml .python-version /app/

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Use non-root user
USER botuser

# Health check (verify Python and imports work)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.config.settings import Settings; print('Health check passed')" || exit 1

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit web UI
# Override with docker run <image> <command> for different modes
CMD ["streamlit", "run", "src/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

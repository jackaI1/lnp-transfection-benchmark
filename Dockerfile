FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python
COPY src ./src
RUN uv sync --locked --no-dev --python /usr/local/bin/python \
    && useradd --uid 10001 --create-home app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "lnp_benchmark.api:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "16"]

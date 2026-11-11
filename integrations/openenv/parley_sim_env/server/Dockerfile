# parley-sim environment for OpenEnv.
#
# Build from the parley-sim repository root (the env installs parley-sim from it
# until parley-sim is on PyPI):
#   docker build -f integrations/openenv/parley_sim_env/server/Dockerfile -t parley-sim-env .
# Run with a scripted customer:
#   docker run -p 8000:8000 parley-sim-env
# Run with an LLM customer (keys are passed at runtime, never baked in):
#   docker run -p 8000:8000 -e OPENROUTER_API_KEY \
#     -e PARLEY_SEATS='{"agent":"client","customer":"model:peer"}' \
#     -e PARLEY_LINEUP='[{"name":"peer","provider":"openai_compat","model":"anthropic/claude-sonnet-5.5","base_url":"https://openrouter.ai/api/v1"}]' \
#     parley-sim-env

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

ENV ENV_DIR=/app/repo/integrations/openenv/parley_sim_env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

COPY . /app/repo
WORKDIR ${ENV_DIR}

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-editable; else uv sync --no-editable; fi

FROM ${BASE_IMAGE}

ENV ENV_DIR=/app/repo/integrations/openenv/parley_sim_env
# Same path as the builder, so the virtualenv's absolute paths stay valid.
COPY --from=builder ${ENV_DIR} ${ENV_DIR}
WORKDIR ${ENV_DIR}

ENV PATH="${ENV_DIR}/.venv/bin:$PATH" \
    PYTHONPATH="${ENV_DIR}" \
    NEMO_TELEMETRY_ENABLED=false

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

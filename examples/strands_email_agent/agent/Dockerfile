FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1

COPY . .

# agentcore-rl-toolkit comes from PyPI via pyproject.toml: `agentcore deploy` builds in
# CodeBuild with a plain `docker build .`, so no extra build context is available.
RUN uv pip install .

# Bake the email database (~650 MB) into the image.
RUN uv run --no-project --with datasets python build_db.py

RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080

CMD ["python", "-m", "rl_app"]

FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ARG PORT=8080
ARG DATASET_DIR=/application_root/data/nba_tracking_data

ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_CACHE_DIR="/application_root/.cache"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/application_root
ENV DATASET_DIR=$DATASET_DIR
ENV PORT=$PORT
ENV VIRTUAL_ENVIRONMENT_PATH="/application_root/.venv"
ENV PATH="$VIRTUAL_ENVIRONMENT_PATH/bin:$PATH"


RUN uv python install 3.12

# Principle of least privilege: create a new user for running the application
RUN groupadd -g 1001 app-user && \
    useradd -r -m -u 1001 -g app-user app-user

# Set the WORKDIR to the application root.
# https://www.uvicorn.org/settings/#development
WORKDIR ${PYTHONPATH}
RUN chown app-user:app-user ${PYTHONPATH}

# Copy the application server configuration
COPY --chown=app-user:app-user gunicorn.conf.py ${PYTHONPATH}

# Document the exposed port
EXPOSE ${PORT}

FROM builder AS development

WORKDIR ${PYTHONPATH}
COPY --chown=app-user:app-user uv.lock pyproject.toml .

RUN uv sync --frozen --no-install-project

# Copy application files
COPY --chown=app-user:app-user src/server ${PYTHONPATH}/server/

CMD ["python", "-m", "server.router.app"]

# Start production stage from builder to avoid inheriting dev tools
FROM builder AS production

WORKDIR ${PYTHONPATH}

COPY --chown=app-user:app-user uv.lock pyproject.toml .

RUN uv sync --frozen --no-install-project --no-dev

# Copy application files
COPY --chown=app-user:app-user src/server ${PYTHONPATH}/server/

# Activate entrypoint for running the Gunicorn application server
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available in Cloud.
CMD ["gunicorn", "--config", "gunicorn.conf.py", "server.router.app:start_server"]
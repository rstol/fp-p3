FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ARG PORT=8080
ARG DATASET_DIR=/application_root/data/nba_tracking_data

ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/application_root
ENV DATASET_DIR=$DATASET_DIR
ENV PORT=$PORT
ENV PATH="/application_root/.venv/bin:$PATH"

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

# Add dev tools and grant sudo access to app-user
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    git \
    curl \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "app-user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/app-user && \
    chmod 0440 /etc/sudoers.d/app-user

WORKDIR /application_root
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
COPY --chown=app-user:app-user . /application_root
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

USER app-user

CMD ["python", "-m", "server.router.app"]

# Start production stage from builder to avoid inheriting dev tools
FROM builder AS production

WORKDIR ${PYTHONPATH}

# Copy only necessary artifacts from development stage
COPY --from=development --chown=app-user:app-user /application_root/.venv /application_root/.venv
COPY --from=development --chown=app-user:app-user /application_root /application_root

# Ensure app-user owns the WORKDIR in production as well
RUN chown app-user:app-user ${PYTHONPATH}
USER app-user

# Activate entrypoint for running the Gunicorn application server
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available in Cloud.
CMD ["gunicorn", "--config", "gunicorn.conf.py", "server.router.app:start_server"]
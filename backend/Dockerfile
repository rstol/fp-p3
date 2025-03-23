# The Poetry installation is provided through the base image. Please check the
# base image if you interested in the details.
# Base image: https://hub.docker.com/r/pfeiffermax/python-poetry
# Dockerfile: https://github.com/max-pfeiffer/python-poetry/blob/main/build/Dockerfile
ARG BASE_IMAGE=pfeiffermax/python-poetry:1.15.0-poetry1.8.5-python3.12.9-slim-bookworm
FROM ${BASE_IMAGE} AS builder
ARG PORT=8080

# https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUNBUFFERED
ENV PYTHONUNBUFFERED=1 \
    # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONDONTWRITEBYTECODE
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/application_root/ \
    # https://python-poetry.org/docs/configuration/#virtualenvsin-project
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR="/application_root/.cache" \
    VIRTUAL_ENVIRONMENT_PATH="/application_root/.venv" \
    PORT=$PORT

# Adding the virtual environment to PATH in order to "activate" it.
# https://docs.python.org/3/library/venv.html#how-venvs-work
ENV PATH="$VIRTUAL_ENVIRONMENT_PATH/bin:$PATH"

# Principle of least privilege: create a new user for running the application
RUN groupadd -g 1001 app-user && \
    useradd -r -u 1001 -g app-user app-user

# Set the WORKDIR to the application root.
# https://www.uvicorn.org/settings/#development
# https://docs.docker.com/engine/reference/builder/#workdir
WORKDIR ${PYTHONPATH}
RUN chown app-user:app-user ${PYTHONPATH}

# Create cache directory and set permissions because user 1001 has no home
# and poetry cache directory.
# https://python-poetry.org/docs/configuration/#cache-directory
RUN mkdir ${POETRY_CACHE_DIR} && chown app-user:app-user ${POETRY_CACHE_DIR}

# Copy the application server configuration
COPY --chown=app-user:app-user gunicorn.conf.py ${PYTHONPATH}

# Document the exposed port
# https://docs.docker.com/engine/reference/builder/#expose
EXPOSE ${PORT}

FROM builder AS development

# install [tool.poetry.dependencies]
# this will install virtual environment into /.venv because of POETRY_VIRTUALENVS_IN_PROJECT=true
# see: https://python-poetry.org/docs/configuration/#virtualenvsin-project
COPY --chown=app-user:app-user poetry.lock pyproject.toml ${PYTHONPATH}
RUN poetry install --no-interaction --no-root --no-ansi

# Copy application files
COPY --chown=app-user:app-user src/server ${PYTHONPATH}server/

CMD ["python", "-m", "server.router.app"]

FROM builder AS production

# install [tool.poetry.dependencies]
# this will install virtual environment into /.venv because of POETRY_VIRTUALENVS_IN_PROJECT=true
# see: https://python-poetry.org/docs/configuration/#virtualenvsin-project
COPY --chown=app-user:app-user poetry.lock pyproject.toml ${PYTHONPATH}
RUN poetry install --no-interaction --no-root --no-ansi --without dev

# Copy application files
COPY --chown=app-user:app-user src/server ${PYTHONPATH}server/

# Activate entrypoint for running the Gunicorn application server
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available in Cloud.
CMD ["gunicorn", "--config", "gunicorn.conf.py", "server.router.app:start_server"]
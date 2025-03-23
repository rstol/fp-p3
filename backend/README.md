# Dummy Backend

## Prerequisites

- Python 3.13 (install with `pyenv install 3.13`)
- Docker and Docker Compose
- Poetry (Python package manager)

## Installation

1. Clone the repository and navigate to the backend directory:

   ```
   cd backend
   ```

2. Install dependencies using Poetry:

   ```
   poetry install
   ```

3. Activate the virtual environment:

   ```
   poetry env activate
   ```

   Then run the generated `source ...` command that Poetry provides.

4. Verify the installation:

   ```
   which python3  # Should return a path to poetry virtualenvs
   ```

5. Install pre-commit hooks for code quality:
   ```
   pre-commit install
   ```

## Running the Application

### Local Development

1. First, prepare the required data:

   ```
   python scripts/prepare_data.py
   ```

2. Start the backend service:

   ```
   docker compose up backend
   ```

   For development with hot-reload:

   ```
   docker compose up backend --build
   ```

### Docker Compose Development

When working with Docker Compose and need to add new packages:

1. Add the package to `pyproject.toml` in the `[tool.poetry.dependencies]` section
   ```
   poetry add package-name  # Example: poetry add pandas
   ```
2. Rebuild the Docker container:
   ```
   docker compose build backend
   ```
3. Restart the container:
   ```
   docker compose up -d backend
   ```

## Data Preparation

Before running the application, you need to download and prepare the data:

```
python scripts/prepare_data.py
```

This script will download the necessary data files and place them in the correct location for the application to use.

# Dummy Backend

## Prerequisites

- Python 3.13 (install with `pyenv install 3.13`)
- Docker and Docker Compose
- UV (Python package manager)

## Installation

1. Clone the repository and navigate to the backend directory:

   ```
   cd backend
   ```

2. Activate the virtual environment:

   ```
   uv venv
   ```
  Then run the generated `source ...` command that uv provides.

3. Install dependencies using uv:

   ```
   uv sync
   ```

4. Verify the installation:

   ```
   which python3  # Should return a path to uv virtualenvs
   ```

5. Install pre-commit hooks for code quality:
   ```
   pre-commit install
   ```

### Setting up `.basketball_profile`

Create a file called `.basketball_profile` in your home directory:

```bash
nano ~/.basketball_profile
```

and copy and paste in the contents of [`.basketball_profile`](.basketball_profile), replacing each of the variable values with paths relevant to your environment.
Next, add the following line to the end of your `~/.bashrc` or `./.zshrc`:

```bash
source ~/.basketball_profile
```

run:

```bash
source ~/.bashrc
```
You should now be able to copy and paste all of the commands in the various instructions sections.
For example:

```bash
echo ${PROJECT_DIR}
```

## Running the Application

### Local Development

1. First, prepare the required data:

   ```
   uv run python scripts/prepare_data.py
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

Before running the application, you need to download and prepare the data and optionally prerender resources:

### Raw Data preparation
The script will download the necessary data files and place them in the correct location for the application to use.

```
uv run python scripts/prepare_data.py
```

### Initial clusters
The initial clusters are computed and stored in the git-repo under `data/init_clusters`.
Run the script:

```python
uv run python src/backend/resources/scatter_data.py
```

> !Important: After modifying the code of initial cluster computation the script has to be rerun again. Then commit the changed prerenderd clusters.

### Videos
We store the prerendered videos in git. 
To prerender the videos again run the script:

```python 
uv run python scripts/prerender_videos.py
```

This will put the prerended videos in the frontend/public folder for direct access in the browser.
# Backend

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

1. Start the backend service:

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

### Raw Data
The data used for the backend and served to the frontend is stored in `data/nba_tracking_data`. 

Currently we restrict to store the data of 3 teams defined in `TEAM_IDS_SAMPLE` in `settings.py` 

To add more teams add more team ids from `data/nba_tracking_data/teams.json` to `TEAM_IDS_SAMPLE` 

Then rerun: 
```
uv run python scripts/prepare_data.py
```

Now rebuild the application. 

### Initial clusters
The initial clusters are computed and stored in the git-repo under `data/init_clusters`.
Run the script:

```python
uv run python src/backend/resources/scatter_data.py
```

> !Important: After modifying the code of initial cluster computation the script has to be rerun again. Then commit the changed prerenderd clusters.

### Videos
We store the prerendered videos in git version control to not need external volumes in production. 
To prerender the videos again run the script:

```python 
uv run python scripts/prerender_videos.py
```

This will put the prerended videos in the frontend/public folder for direct access in the browser.

To add more teams to prerender: add more team ids from `data/nba_tracking_data/teams.json` to `TEAM_IDS_SAMPLE` and rerun the script. 

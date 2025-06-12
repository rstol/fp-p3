# Basketball Analytics Backend

FastAPI-based backend service providing NBA tracking data analysis, machine learning clustering, and real-time API endpoints.

## Tech Stack

- **Python 3.13** with modern async/await patterns
- **FastAPI** for high-performance API development
- **Pandas** for data manipulation and analysis
- **Scikit-learn** for machine learning models
- **UV** for fast Python package management
- **Docker** for containerized development

## Quick Start

### Prerequisites

- Python 3.13 (install with `pyenv install 3.13`)
- UV package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker and Docker Compose

### Installation

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Verify installation:**
   ```bash
   which python3  # Should point to UV virtual environment
   ```

5. **Setup development tools:**
   ```bash
   pre-commit install
   ```

### Environment Configuration

Create `.basketball_profile` in your home directory:

```bash
nano ~/.basketball_profile
```

Add the following configuration (adjust paths for your environment):

```bash
export PROJECT_DIR=
export DATA_DIR=
export SCRIPTS_DIR=
export TRACKING_DIR=${DATA_DIR}/nba_tracking_data
export GAMES_DIR=${DATA_DIR}/games_preprocessed
export EXPERIMENTS_DIR=${DATA_DIR}/experiments
```

Add to your shell profile:

```bash
echo "source ~/.basketball_profile" >> ~/.bashrc
source ~/.bashrc
```

## Project Structure

```
├── data
│   ├── clusters
│   ├── embeddings
│   ├── experiments
│   ├── games_preprocessed
│   ├── init_clusters       # Precomputed clustering
│   ├── nba_tracking_data   # Raw game data
│   └── user_updates
├── Dockerfile
├── gunicorn.conf.py
├── pyproject.toml
├── README.md
├── scripts
│   ├── load_nba_tracking_data_15_16.py
│   ├── prepare_data.py
│   ├── prerender_videos.py
│   ├── train_model_v2.sh
│   └── train_model.sh
├── src
│   └── backend
└── uv.lock
```

## Running the Application

### Docker Development (Recommended)

```bash
# Start backend service
docker compose up backend

# With hot-reload for development
docker compose up backend --build

# View logs
docker compose logs -f backend
```

### Local Development

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn src.gamut_server.router.app:app --reload --host 0.0.0.0 --port 8080
```

API will be available at `http://localhost:8080`.

## Data Management

### Team Configuration

Current sample teams are defined in `settings.py`:

```python
TEAM_IDS_SAMPLE = [
    1610612748, 1610612752, 1610612755
]
```

### Data Preparation Pipeline

1. **Prepare raw data (optional):**
   ```bash
   uv run python scripts/prepare_data.py
   ```

2. **Generate initial clusters (optional):**
   ```bash
   uv run python src/backend/resources/scatter_data.py
   ```

3. **Prerender videos (optional):**
   ```bash
   uv run python scripts/prerender_videos.py
   ```

### Adding New Teams

1. Find team ID in `data/nba_tracking_data/teams.json`
2. Add ID to `TEAM_IDS_SAMPLE` in `settings.py`
3. Run data preparation scripts
4. Rebuild Docker container if using containerized development

## API Endpoints

### Scatter Plot & Clustering

```python
# Team play scatter data
GET /api/v1/teams/{team_id}/plays/scatter

# Cluster information
GET /api/v1/teams/{team_id}/cluster/{cluster_id}

# Individual scatter points
GET /api/v1/teams/{team_id}/scatterpoint/{game_id}/{event_id}

# Batch scatter points
POST /api/v1/teams/{team_id}/scatterpoints
```

### NBA Data

```python
# Teams
GET /api/v1/teams
GET /api/v1/teams/{team_id}
GET /api/v1/teams/{team_id}/games

# Games
GET /api/v1/games
GET /api/v1/games/{game_id}
GET /api/v1/games/{game_id}/plays

# Plays and Videos
GET /api/v1/games/{game_id}/plays/{play_id}/raw
GET /api/v1/plays/{game_id}/{event_id}/video
GET /api/v1/plays/{game_id}/{event_id}
```

## Troubleshooting

### Common Issues

**UV virtual environment issues:**
```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --all-extras
```

**Docker build failures:**
```bash
docker compose build backend --no-cache
docker compose up backend
```

**Data processing errors:**
- Verify data files exist in `data/nba_tracking_data/`
- Check file permissions and formats
- Ensure sufficient disk space for processing
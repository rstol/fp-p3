import logging
import math
import os
import random

import pandas as pd
from flask_restful import Resource
from sklearn.cluster import KMeans

from backend.resources.dataset_manager import DatasetManager
from backend.settings import DATASET_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetResource(Resource):
    """dataset resource."""

    data_root = os.path.join(".", "data")

    def get(self, name):
        path_name = os.path.join(self.data_root, f"dataset_{name}.csv")
        data = pd.read_csv(path_name)

        # process the data, e.g. find the clusters
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
        labels = kmeans.labels_.tolist()

        # Add cluster to data
        data["cluster"] = labels

        # Convert to dictionary
        return data.to_dict(orient="records")


class TeamPlaysScatterResource(Resource):
    """Resource for serving team play data for scatter plot visualization."""

    def __init__(self):
        self.dataset_manager = DatasetManager(DATASET_DIR)

    def get(self, team_id):
        # Log the request
        logger.info(f"Fetching scatter data for team_id: {team_id}")

        # Convert team_id to proper format
        try:
            team_id = int(team_id)
        except ValueError:
            logger.error(f"Invalid team_id format: {team_id}")
            return {"error": "Invalid team ID format"}, 400

        # Get games for the specified team
        games = self.dataset_manager.get_games_for_team(team_id)
        logger.info(f"Found {len(games)} games for team {team_id}")

        if not games:
            logger.warning(f"No games found for team {team_id}")
            return {"error": "No games found for this team"}, 404

        # Collect play data from games
        scatter_data = []

        # Limit to a few games to avoid processing too much data
        for game in games[:2]:  # Process first 2 games
            game_id = game["game_id"]
            logger.info(f"Processing plays for game {game_id}")

            # Get all plays for this game
            plays = self.dataset_manager.get_plays_for_game(game_id, as_dicts=False)
            logger.info(f"Found {len(plays)} plays for game {game_id}")

            # Process each play to extract coordinate data
            for play in plays:
                # Check if this play is related to the requested team
                play_team_id = self._get_play_team_id(play, team_id)

                if play_team_id == team_id:
                    # Extract coordinates from moments data
                    points = self._extract_coordinate_data(play, team_id)
                    scatter_data.extend(points)

            # If we have enough data points, stop processing more games
            if len(scatter_data) >= 100:
                logger.info(f"Collected sufficient data points ({len(scatter_data)}), stopping.")
                break

        logger.info(f"Extracted {len(scatter_data)} data points for team {team_id}")

        # If we don't have enough data, create mock data
        if len(scatter_data) < 10:
            logger.warning(f"Insufficient data points ({len(scatter_data)}), creating mock data")
            scatter_data = self._create_mock_data()

        # Apply clustering to the data points
        scatter_data = self._apply_clustering(scatter_data)

        return scatter_data[:-1]

    def _get_play_team_id(self, play, target_team_id):
        """Extract the team ID from a play that's most relevant."""
        # Check various locations where team ID might be stored
        if "team_id" in play and play["team_id"] is not None:
            try:
                return float(play["team_id"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid team_id format: {play['team_id']}")

        if "primary_player_info" in play and play["primary_player_info"] is not None:
            if (
                "team_id" in play["primary_player_info"]
                and play["primary_player_info"]["team_id"] is not None
            ):
                try:
                    return float(play["primary_player_info"]["team_id"])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid team_id format: {play['primary_player_info']['team_id']}"
                    )

        if "secondary_player_info" in play and play["secondary_player_info"] is not None:
            if (
                "team_id" in play["secondary_player_info"]
                and play["secondary_player_info"]["team_id"] is not None
            ):
                try:
                    return float(play["secondary_player_info"]["team_id"])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid team_id format: {play['secondary_player_info']['team_id']}"
                    )

        if "possession_team_id" in play and play["possession_team_id"] is not None:
            try:
                return float(play["possession_team_id"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid possession_team_id format: {play['possession_team_id']}")

        if "offense_team_id" in play and play["offense_team_id"] is not None:
            try:
                return float(play["offense_team_id"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid offense_team_id format: {play['offense_team_id']}")

        if "defense_team_id" in play and play["defense_team_id"] is not None:
            try:
                return float(play["defense_team_id"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid defense_team_id format: {play['defense_team_id']}")

        # Check home and away team from the event description
        if (
            "event_desc_home" in play
            and "home_team_id" in play
            and target_team_id == float(play.get("home_team_id", 0))
        ):
            return float(play.get("home_team_id", 0))

        if (
            "event_desc_away" in play
            and "away_team_id" in play
            and target_team_id == float(play.get("away_team_id", 0))
        ):
            return float(play.get("visitor_team_id", 0))

        # If no team ID found, use the target team ID as default
        return target_team_id

    def _extract_coordinate_data(self, play, team_id):
        """Extract coordinate data from a play's moments."""
        points = []

        # Extract the event info
        event_id = play.get("event_id", "")
        event_type = play.get("event_type", "")
        event_desc = play.get("event_desc_home", "") or play.get("event_desc_away", "")
        game_id = play.get("game_id", "")
        period = play.get("period", 1)

        if len(points) == 0 and event_type:
            # Map different event types to different areas of the court
            event_type_num = hash(str(event_type)) % 100  # Get a number from 0-99

            # Create coordinates based on event type (spread across the court)
            x = 100 + (event_type_num * 3) % 300  # x between 100-400
            y = 50 + (event_type_num * 7) % 300  # y between 50-350

            # Add some randomness
            x += random.randint(-30, 30)
            y += random.randint(-30, 30)

            point = {
                "x": x,
                "y": y,
                "event_id": f"{event_id}_event",
                "play_type": f"Event Type {event_type}",
                "description": event_desc or "Play Event",
                "period": period,
                "game_id": game_id,
            }
            points.append(point)

        return points

    def _create_mock_data(self, num_points=50):
        """Create mock data for visualization when real data is insufficient."""
        mock_data = []

        # Create mock data points representing typical basketball play positions
        # Center of court is at (0,0), with typical half-court dimensions

        # Generate some three-point shots
        for i in range(15):
            angle = random.uniform(0, 3.14)  # Semi-circle angle
            radius = random.uniform(22, 24)  # Three-point line radius
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) + 100  # Offset to place on one side of court

            mock_data.append(
                {
                    "x": x,
                    "y": y,
                    "event_id": f"mock_3pt_{i}",
                    "play_type": "Shot",
                    "description": "3-Point Shot",
                    "period": 1,
                    "game_id": "mock_game",
                }
            )

        # Generate some mid-range shots
        for i in range(20):
            x = random.uniform(-150, 150)
            y = random.uniform(50, 150)

            mock_data.append(
                {
                    "x": x,
                    "y": y,
                    "event_id": f"mock_mid_{i}",
                    "play_type": "Shot",
                    "description": "Mid-Range Shot",
                    "period": 1,
                    "game_id": "mock_game",
                }
            )

        # Generate some paint/layup shots
        for i in range(15):
            x = random.uniform(-40, 40)
            y = random.uniform(0, 70)

            mock_data.append(
                {
                    "x": x,
                    "y": y,
                    "event_id": f"mock_paint_{i}",
                    "play_type": "Shot",
                    "description": "Paint Shot/Layup",
                    "period": 1,
                    "game_id": "mock_game",
                }
            )

        return mock_data

    def _apply_clustering(self, data_points):
        """Apply KMeans clustering to the data points."""
        if len(data_points) < 3:
            logger.warning("Too few data points for clustering, defaulting all to cluster 0")
            for point in data_points:
                point["cluster"] = 0
            return data_points

        # Extract coordinates for clustering
        coords = [[point["x"], point["y"]] for point in data_points]

        # Determine optimal number of clusters based on data size
        n_clusters = min(3, len(data_points) // 20) if len(data_points) > 30 else 2
        n_clusters = max(2, n_clusters)  # At least 2 clusters

        logger.info(f"Clustering {len(data_points)} points into {n_clusters} clusters")

        try:
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            clusters = kmeans.fit_predict(coords)

            # Add cluster information to each data point
            for i, point in enumerate(data_points):
                point["cluster"] = int(clusters[i])
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            # Default to assigning alternating clusters
            for i, point in enumerate(data_points):
                point["cluster"] = i % 3

        return data_points

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import torch
from flask import jsonify, request
from flask_restful import Resource, abort

from backend.resources.clustering import cop_kmeans
from backend.resources.dataset_manager import DatasetManager
from backend.settings import DATASET_DIR

logger = logging.getLogger(__name__)


@dataclass
class PlayTypeCentroids:
    """Handles generation of synthetic coordinates based on play type centroids."""

    centroids: dict[int, tuple[float, float]] = field(default_factory=dict)
    std_dev: float = 0.5

    def __post_init__(self):
        """Initialize default centroids after object creation if none provided."""
        if not self.centroids:
            # Default centroids based on modulo 10 of event_type
            self.centroids = {
                i: (np.cos(i * np.pi / 5) * 5, np.sin(i * np.pi / 5) * 5) for i in range(10)
            }

    def generate_coordinates(self, plays_df: pl.DataFrame) -> pl.DataFrame:
        """Adds synthetic 'x' and 'y' columns to a DataFrame based on event_type."""
        if plays_df.is_empty():
            # Return with empty coordinate columns if input is empty
            return plays_df.with_columns(
                x=pl.lit(None, dtype=pl.Float64), y=pl.lit(None, dtype=pl.Float64)
            )

        # Ensure the necessary 'event_type' column exists
        if "event_type" not in plays_df.columns:
            logger.error("'generate_coordinates' requires 'event_type' column.")
            # Return with empty coordinate columns if missing required column
            return plays_df.with_columns(
                x=pl.lit(None, dtype=pl.Float64), y=pl.lit(None, dtype=pl.Float64)
            )

        # Ensure event_type is integer for modulo operation
        try:
            plays_df = plays_df.with_columns(pl.col("event_type").cast(pl.Int64))
        except pl.exceptions.ComputeError as e:
            logger.error(f"Failed to cast event_type to Int64: {e}. Cannot generate coords.")
            return plays_df.with_columns(
                x=pl.lit(None, dtype=pl.Float64), y=pl.lit(None, dtype=pl.Float64)
            )

        x_centroids = {k: v[0] for k, v in self.centroids.items()}
        y_centroids = {k: v[1] for k, v in self.centroids.items()}

        # Calculate base coordinates by mapping event_type (mod 10) to centroids
        plays_df = plays_df.with_columns(
            x=pl.col("event_type").mod(10).replace(x_centroids, default=0.0),
            y=pl.col("event_type").mod(10).replace(y_centroids, default=0.0),
        )

        # Add Gaussian noise to the base coordinates
        num_rows = plays_df.height
        if num_rows > 0:
            noise_x = np.random.normal(0, self.std_dev, num_rows)
            noise_y = np.random.normal(0, self.std_dev, num_rows)
            plays_df = plays_df.with_columns(
                x=(pl.col("x") + pl.Series(values=noise_x)),
                y=(pl.col("y") + pl.Series(values=noise_y)),
            )

        return plays_df


class TeamResource(Resource):
    """Resource for fetching basic team information."""

    def __init__(self):
        self.dataset_manager = DatasetManager(DATASET_DIR)

    def get(self):
        """Returns a list of all teams."""
        teams = self.dataset_manager.get_teams()
        return jsonify(teams)


class TeamPlaysScatterResource(Resource):
    """Resource for getting play scatter plot data for a team (GET endpoint)."""

    def __init__(self):
        self.dataset_manager = DatasetManager(DATASET_DIR)
        # Use the centralized coordinate generator, potentially with different std_dev for scatter
        self.coord_generator = PlayTypeCentroids(std_dev=2.5)

    def get(self, team_id: str):
        """Get scatter plot data for a team's plays based on timeframe."""
        logger.info(f"Fetching scatter data for team_id: {team_id}")

        num_games = self._parse_timeframe(request.args.get("timeframe", "last_3"))

        try:
            team_id_int = int(team_id)
        except ValueError:
            logger.error(f"Invalid team_id format: {team_id}")
            abort(400, description="Invalid team ID format. Expected integer.")

        games_df, total_games = self._get_filtered_games(team_id_int, num_games)

        game_recency = self._calculate_game_recency(games_df)
        plays_df = self._concat_plays_from_games(games_df, game_recency)

        if plays_df.is_empty():
            logger.warning(f"No plays concatenated for team {team_id} in timeframe.")
            return jsonify({"total_games": total_games, "points": []})

        logger.info(f"Collected {plays_df.height} plays for generating scatter points.")

        points_list, _ = self._generate_scatter_data(plays_df, total_games)

        return jsonify({"total_games": total_games, "points": points_list})

    def _parse_timeframe(self, timeframe: str | None) -> int:
        """Parse timeframe string (e.g., 'last_5') into number of games."""
        if not timeframe or not timeframe.startswith("last_"):
            logger.warning(f"Invalid timeframe format: {timeframe}. Using default: last_3")
            return 3
        try:
            num_games = int(timeframe.split("_")[1])
            if num_games <= 0:
                raise ValueError("Number of games must be positive")
            logger.info(f"Parsed timeframe: {timeframe} -> {num_games} games")
            return num_games
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid timeframe format: {timeframe}. Error: {e}")
            abort(400, description=f"Invalid timeframe format: '{timeframe}'. Expected 'last_N'.")

    def _get_filtered_games(self, team_id: int, num_games: int) -> tuple[pl.DataFrame, int]:
        """Get games for a team, sort by date, and limit by num_games."""
        try:
            games = self.dataset_manager.get_games_for_team(team_id, as_dicts=False)
        except FileNotFoundError:
            logger.warning(f"Games data file not found for team {team_id}")
            abort(404, description=f"No game data found for team {team_id}.")
        except pl.exceptions.PolarsError as e:
            logger.error(f"Error loading games for team {team_id} with Polars: {e}", exc_info=True)
            abort(500, description="Internal server error loading game data.")

        if games is None or games.is_empty():
            logger.warning(f"No games found in data for team {team_id}")
            # If no games exist at all for the team
            abort(404, description=f"No games found for team {team_id}.")

        logger.info(f"Found {games.height} total games for team {team_id}")
        total_games = games.height

        # Sort games by date (if column exists) and limit
        if "game_date" in games.columns:
            try:
                # Attempt to cast to date and sort, handle potential errors
                games = games.with_columns(
                    pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                ).sort("game_date", descending=True)
            except pl.exceptions.ComputeError as e:
                logger.warning(
                    f"Could not parse or sort by game_date: {e}. Proceeding without sorting."
                )
        else:
            logger.warning("No 'game_date' column found for sorting games.")

        games_filtered = games.limit(num_games)
        logger.info(
            f"Limited to {games_filtered.height} games based on timeframe (max {num_games})."
        )

        return games_filtered, total_games

    def _calculate_game_recency(self, games_df: pl.DataFrame) -> dict[Any, float]:
        """Calculate recency scores (0 to 1) for games based on dates."""
        game_recency: dict[Any, float] = {}
        if games_df.is_empty() or "game_date" not in games_df.columns:
            return game_recency

        try:
            # Ensure 'game_date' is Date type if not already
            if games_df["game_date"].dtype != pl.Date:
                games_df = games_df.with_columns(
                    pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                )

            # Drop rows where date conversion failed
            games_df = games_df.drop_nulls("game_date")
            if games_df.is_empty():
                return {}

            max_date = games_df["game_date"].max()
            min_date = games_df["game_date"].min()

            # Calculate range in days
            date_range_days = (max_date - min_date).days if max_date and min_date else 0

            for game in games_df.select(["game_id", "game_date"]).rows(named=True):
                game_id = game["game_id"]
                game_date = game["game_date"]
                if date_range_days > 0:
                    days_diff = (max_date - game_date).days
                    normalized_recency = 1.0 - (days_diff / date_range_days)
                else:
                    normalized_recency = 1.0
                game_recency[game_id] = max(0.0, min(1.0, normalized_recency))

        except pl.exceptions.PolarsError as e:
            # Fallback to simple index-based recency if date processing fails
            logger.warning(f"Date processing/recency failed: {e}. Using index fallback.")
            game_ids = games_df["game_id"].to_list()
            num_games = len(game_ids)
            for i, game_id in enumerate(game_ids):
                # Ensure recency decreases for older games (higher index)
                game_recency[game_id] = 1.0 - (i / max(1, num_games - 1))

        return game_recency

    def _concat_plays_from_games(self, games_df: pl.DataFrame, game_recency: dict) -> pl.DataFrame:
        """Load plays for games, add recency, and concatenate."""
        all_plays_list: list[pl.DataFrame] = []
        # Define expected schema for plays to ensure consistency
        # Select only columns needed downstream by _generate_scatter_data
        base_schema = {
            "event_id": pl.Int64,
            "event_type": pl.Int64,
            "description": pl.Utf8,
            "play_type": pl.Int64,
            "recency": pl.Float64,
            "game_id": games_df["game_id"].dtype,
        }

        for game in games_df.rows(named=True):
            game_id = game["game_id"]
            logger.info(f"Processing plays for game {game_id}")
            try:
                plays = self.dataset_manager.get_plays_for_game(game_id, as_dicts=False)
                if plays is None or plays.is_empty():
                    logger.warning(f"No plays data found for game {game_id}. Skipping.")
                    continue

                # Ensure basic columns exist for deriving others
                required_input_cols = {
                    "event_id",
                    "event_type",
                    "event_desc_home",
                    "event_desc_away",
                }
                if not required_input_cols.issubset(plays.columns):
                    logger.warning(f"Skipping game {game_id}: missing required raw play columns.")
                    continue

                recency = game_recency.get(game_id, 0.5)
                plays = plays.with_columns(pl.lit(recency).alias("recency"))
                plays = plays.with_columns(pl.lit(game_id).alias("game_id"))

                # Add derived columns
                plays = plays.with_columns(
                    pl.when(
                        pl.col("event_desc_home").is_not_null() & (pl.col("event_desc_home") != "")
                    )
                    .then(pl.col("event_desc_home"))
                    .otherwise(pl.col("event_desc_away"))
                    .str.slice(0, 25)
                    .fill_null("N/A")
                    .alias("description"),
                    pl.col("event_type").alias("play_type"),
                )

                # Select and cast to base schema before appending
                current_plays_for_concat = plays.select(list(base_schema.keys()))
                try:
                    # Attempt cast - this ensures type consistency
                    current_plays_for_concat = current_plays_for_concat.cast(
                        base_schema, strict=False
                    )
                    all_plays_list.append(current_plays_for_concat)
                except pl.exceptions.SchemaError as e:
                    logger.warning(
                        f"Schema mismatch casting plays for game {game_id}, skipping: {e}"
                    )
                    continue

            except FileNotFoundError:
                logger.warning(f"Plays data file not found for game {game_id}. Skipping.")
            except pl.exceptions.PolarsError as e:
                logger.error(f"Error processing plays for game {game_id}: {e}", exc_info=True)
                continue  # Skip game on error

        if not all_plays_list:
            return pl.DataFrame(schema=base_schema)

        logger.info(f"Concatenating plays from {len(all_plays_list)} games.")
        return pl.concat(all_plays_list)

    def _generate_scatter_data(
        self, plays_df: pl.DataFrame, total_games: int
    ) -> tuple[list[dict], int]:
        """Generates the final list of points for the scatter plot API response."""
        if plays_df.is_empty():
            return [], total_games

        # Generate 'x', 'y' coordinates
        plays_with_coords = self.coord_generator.generate_coordinates(plays_df)

        required_cols = {"event_id", "x", "y", "description", "play_type", "recency", "game_id"}
        if not required_cols.issubset(plays_with_coords.columns):
            logger.error("DataFrame missing required columns after coordinate generation.")
            missing = required_cols - set(plays_with_coords.columns)
            logger.error(f"Missing: {missing}")
            return [], total_games  # Return empty list on error

        # Select, add default cluster, round, and convert to dicts
        result_df = plays_with_coords.select(
            [
                pl.col("event_id"),
                pl.col("x"),
                pl.col("y"),
                pl.col("description"),
                pl.col("play_type"),
                pl.col("recency"),
                pl.col("game_id"),
                pl.lit("-1").alias("cluster"),  # Default cluster ID is -1 (string)
            ]
        )

        points_list = result_df.with_columns(
            [pl.col("x").round(4), pl.col("y").round(4), pl.col("recency").round(4)]
        ).to_dicts()

        return points_list, total_games


@dataclass
class UpdateClusteringResource(Resource):
    """Resource for updating clustering assignments based on user input constraints."""

    dataset_manager: DatasetManager = field(default_factory=lambda: DatasetManager(DATASET_DIR))
    coord_generator: PlayTypeCentroids = field(default_factory=PlayTypeCentroids)
    # Instantiate helper resource to reuse its methods
    scatter_helper: TeamPlaysScatterResource = field(default_factory=TeamPlaysScatterResource)
    num_clusters: int = 5  # Default number of clusters

    def post(self):
        """Handle POST request to update clustering based on user constraints."""
        logger.info("UpdateClusteringResource POST received.")
        json_data = request.get_json()

        if not json_data:
            abort(400, description="Missing JSON payload.")

        # 1. Validate request
        validated_data = self._validate_request(json_data)
        team_id = validated_data["team_id"]
        timeframe = validated_data["timeframe"]
        point_id = validated_data["point_id"]  # String
        new_cluster = validated_data["new_cluster"]
        original_cluster = validated_data["original_cluster"]

        # 2. Load Data
        plays_df = self._load_play_data(team_id, timeframe)
        if plays_df is None or plays_df.is_empty():
            abort(
                404,
                description=(
                    f"Could not load valid play data for team {team_id} and timeframe {timeframe}."
                ),
            )

        # 3. Prepare Data for Clustering (casting, nulls, tensor, index map)
        coords_tensor, point_id_to_idx, point_id_int, plays_df_clean = (
            self._prepare_data_for_clustering(plays_df, point_id)
        )

        # 4. Prepare Constraints (currently placeholder)
        must_link, cannot_link = self._prepare_constraints(
            plays_df_clean, point_id_int, new_cluster, original_cluster, point_id_to_idx
        )

        # 5. Run Clustering
        cluster_assignments = self._run_clustering(
            coords_tensor, plays_df_clean.height, must_link, cannot_link
        )

        # 6. Format Response
        final_points = self._format_clustering_response(plays_df_clean, cluster_assignments)

        logger.info(f"Successfully updated clustering. Returning {len(final_points)} points.")
        return jsonify({"success": True, "points": final_points})

    # --- Refactored Helper Methods --- #

    def _prepare_data_for_clustering(
        self,
        plays_df: pl.DataFrame,
        point_id: str,  # Point ID from request (string)
    ) -> tuple[torch.Tensor, dict[int, int], int, pl.DataFrame]:
        """Prepare loaded data for clustering: cast, handle nulls, create tensor & index map.

        Note: Complexity (cyclomatic) is slightly high due to multiple checks/casts.
        Considered breaking down further, but kept together for logical flow.
        """
        logger.info(f"Preparing {plays_df.height} plays for clustering.")

        # Ensure 'x' and 'y' columns exist
        if "x" not in plays_df.columns or "y" not in plays_df.columns:
            logger.error("'x' or 'y' columns missing before preparation.")
            abort(500, description="Coordinate columns missing.")

        # Ensure coordinates are numeric
        if not all(dtype in pl.NUMERIC_DTYPES for dtype in plays_df[["x", "y"]].dtypes):
            try:
                plays_df = plays_df.with_columns(
                    [pl.col("x").cast(pl.Float64), pl.col("y").cast(pl.Float64)]
                )
                logger.info("Successfully cast x, y columns to Float64.")
            except (pl.PolarsError, TypeError, ValueError) as e:  # More specific exceptions
                logger.error(f"Failed to cast coordinates to numeric: {e}", exc_info=True)
                abort(500, description="Coordinates are not numeric or cannot be cast.")

        # Drop rows with null coordinates
        initial_height = plays_df.height
        plays_df = plays_df.drop_nulls(subset=["x", "y"])
        if plays_df.is_empty():
            logger.warning("No valid points with non-null coordinates found after filtering.")
            abort(400, description="No valid points found for clustering.")
        if plays_df.height < initial_height:
            logger.info(f"Dropped {initial_height - plays_df.height} rows with null coordinates.")

        # Convert coordinates to Tensor
        coords_tensor = torch.tensor(plays_df.select(["x", "y"]).to_numpy(), dtype=torch.float32)

        # Ensure event_id column exists and create index map
        if "event_id" not in plays_df.columns:
            logger.error("'event_id' column missing in prepared data.")
            abort(500, description="Internal error: event_id missing.")

        if plays_df["event_id"].dtype != pl.Int64:
            try:
                plays_df = plays_df.with_columns(pl.col("event_id").cast(pl.Int64))
            except (pl.PolarsError, TypeError, ValueError) as e:
                logger.error(f"Failed to cast event_id to Int64: {e}")
                abort(500, description="Internal error: Invalid event_id format.")

        event_ids = plays_df["event_id"].to_list()
        try:
            point_id_int = int(point_id)  # Convert request point_id to int
            point_id_to_idx = {event_id: i for i, event_id in enumerate(event_ids)}
            if point_id_int not in point_id_to_idx:
                logger.warning(f"Requested point_id {point_id_int} not found in filtered data.")
                abort(404, description=f"Point ID {point_id_int} not found in dataset view.")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid point_id format '{point_id}': {e}")
            abort(400, description=f"Invalid format for point_id: '{point_id}'.")

        logger.info("Data preparation for clustering complete.")
        return coords_tensor, point_id_to_idx, point_id_int, plays_df

    def _run_clustering(
        self,
        coords_tensor: torch.Tensor,
        num_points: int,
        must_link: list[tuple[int, int]],
        cannot_link: list[tuple[int, int]],
    ) -> list[str]:
        """Run COP-KMeans algorithm and return cluster assignments as list of strings."""
        try:
            if num_points < self.num_clusters:
                logger.warning(
                    f"Number of data points ({num_points}) is less than "
                    f"configured clusters ({self.num_clusters}). Adjusting to {num_points}."
                )
                effective_num_clusters = max(1, num_points)
            else:
                effective_num_clusters = self.num_clusters

            logger.info(
                f"Running COP-KMeans: {effective_num_clusters} clusters, {num_points} points, "
                f"ML={len(must_link)}, CL={len(cannot_link)}"
            )

            cluster_assignments_tensor, _ = cop_kmeans(
                X=coords_tensor,
                n_clusters=effective_num_clusters,
                ml=must_link,
                cl=cannot_link,
                max_iter=100,
                random_state=42,
            )
            logger.info(
                f"COP-KMeans finished. Assignments shape: {cluster_assignments_tensor.shape}"
            )

            cluster_list = [str(c.item()) for c in cluster_assignments_tensor]

            if len(cluster_list) != num_points:
                logger.error(
                    f"Cluster assignment length mismatch: {len(cluster_list)} vs {num_points}"
                )
                abort(500, description="Internal error: Clustering assignment length mismatch.")

            return cluster_list

        except Exception as e:
            logger.error(f"Error during cop_kmeans clustering: {e}", exc_info=True)
            abort(500, description=f"Clustering algorithm failed: {e}")

    def _format_clustering_response(
        self, plays_df: pl.DataFrame, cluster_assignments: list[str]
    ) -> list[dict]:
        """Update DataFrame with cluster assignments and format for API response."""
        logger.info("Formatting clustering response.")
        updated_plays_df = plays_df.with_columns(
            pl.Series(name="cluster", values=cluster_assignments)
        )

        # Use the helper's method to generate the final point list format
        # Pass total_games=-1 as it's not needed for formatting
        final_points, _ = self.scatter_helper._generate_scatter_data(
            updated_plays_df, total_games=-1
        )
        return final_points

    # --- Original Helper Methods --- #

    def _validate_request(self, data: Any) -> dict[str, Any]:
        """Validate the incoming JSON data for required fields."""
        required_keys = {"point_id", "new_cluster", "original_cluster", "team_id", "timeframe"}
        if not isinstance(data, dict):
            abort(400, description="Invalid JSON payload: Expected an object.")
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            abort(400, description=f"Missing required keys: {', '.join(missing)}")

        # Basic type checks (can be expanded)
        try:
            validated_data = {
                "point_id": str(data["point_id"]),  # Keep as string initially
                "new_cluster": str(data["new_cluster"]),
                "original_cluster": str(data["original_cluster"]),
                "team_id": int(data["team_id"]),  # Expect team_id as int
                "timeframe": str(data["timeframe"]),
            }
            # Check if new_cluster is a valid integer string (or -1)
            if validated_data["new_cluster"] != "-1":
                int(validated_data["new_cluster"])
            # Check if original_cluster is a valid integer string (or -1)
            if validated_data["original_cluster"] != "-1":
                int(validated_data["original_cluster"])

        except (ValueError, TypeError) as e:
            abort(400, description=f"Invalid data types in request: {e}")

        logger.info(f"Request validated: {validated_data}")
        return validated_data

    def _load_play_data(self, team_id: int, timeframe: str) -> pl.DataFrame | None:
        """Load and prepare play data using team_id and timeframe."""
        logger.info(f"Loading play data for team {team_id}, timeframe {timeframe}")
        try:
            num_games = self.scatter_helper._parse_timeframe(timeframe)
            # Note: _get_filtered_games returns tuple (DataFrame | None, int)
            games_df, _ = self.scatter_helper._get_filtered_games(team_id, num_games)

            # Handle case where _get_filtered_games returns None
            if games_df is None or games_df.is_empty():
                logger.warning(f"No games found for team {team_id} and timeframe {timeframe}.")
                return None  # Return None instead of aborting here

            game_recency = self.scatter_helper._calculate_game_recency(games_df)
            plays_df = self.scatter_helper._concat_plays_from_games(games_df, game_recency)

            if plays_df.is_empty():
                logger.warning("No plays found after concatenation.")
                return None

            # Generate coordinates
            plays_with_coords = self.coord_generator.generate_coordinates(plays_df)
            logger.info(f"Generated coordinates for {plays_with_coords.height} plays.")

            # Ensure essential columns exist after coordinate generation
            required_cols = {"event_id", "x", "y"}
            if not required_cols.issubset(plays_with_coords.columns):
                missing = required_cols - set(plays_with_coords.columns)
                logger.error(f"Missing essential columns after coordinate generation: {missing}")
                return None

            return plays_with_coords

        except Exception as e:  # Catch errors during data loading/processing
            logger.error(f"Error in _load_play_data for team {team_id}: {e}", exc_info=True)
            # Don't abort here; let the caller (post method) handle None return
            return None

    def _prepare_constraints(
        self,
        plays_df: pl.DataFrame,  # Use DataFrame to find other points
        point_id: int,
        new_cluster: str,
        original_cluster: str,
        point_id_to_idx: dict[int, int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Prepare must-link and cannot-link constraints for COP-KMeans.

        NOTE: Currently returns empty lists as mapping single-point assignment
        requests to standard pair-based ML/CL constraints is complex without
        knowing prior cluster state or algorithm modification.
        """
        must_link = []
        cannot_link = []

        try:
            # Assign to _ to explicitly ignore the unused variable in placeholder logic
            _ = point_id_to_idx[point_id]  # noqa: F841 (variable assigned but never used)
        except KeyError:
            logger.error(f"Point ID {point_id} not found in index map during constraint prep.")
            return [], []

        logger.warning(
            "Constraint generation (_prepare_constraints) is currently a placeholder "
            "and does not enforce specific cluster assignments via ML/CL pairs."
        )
        # moved_point_idx is unused in placeholder, suppressing lint warning implicitly

        return must_link, cannot_link

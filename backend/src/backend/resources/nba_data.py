import polars as pl
from flask import Response, jsonify, request
from flask_restful import Resource
from loguru import logger

from backend.resources.dataset_manager import DatasetManager
from backend.settings import DATA_DIR, TRACKING_DIR, UPDATE_PLAY_SCHEMA

dataset_manager = DatasetManager(TRACKING_DIR)


class TeamsResource(Resource):
    def get(self):
        return jsonify(dataset_manager.get_teams())


class TeamDetailsResource(Resource):
    def get(self, team_id):
        team = dataset_manager.get_team_details(team_id)
        if team:
            return jsonify(team)
        return {"error": "Team not found"}, 404


class TeamGamesResource(Resource):
    def get(self, team_id):
        return jsonify(dataset_manager.get_games_for_team(team_id))


class GamesResource(Resource):
    def get(self):
        return dataset_manager.get_games()


class GameDetailsResource(Resource):
    def get(self, game_id):
        game = dataset_manager.get_game_details(game_id)
        if game:
            return jsonify(game)
        return {"error": "Game not found"}, 404


class GamePlaysResource(Resource):
    def get(self, game_id):
        return jsonify(dataset_manager.get_plays_for_game(game_id))


class PlayRawDataResource(Resource):
    def get(self, game_id, play_id):
        try:
            play = dataset_manager.get_play_raw_data(game_id, play_id)
            if play:
                return jsonify(play)
        except ValueError:
            return {"error": "Invalid play ID format"}, 400

        return {"error": "Play not found"}, 404


class PlayVideoResource(Resource):
    def get(self, game_id, event_id):
        try:
            video = dataset_manager.get_play_video(game_id, event_id)
            if video:
                return Response(video, mimetype="application/octet-stream")
        except ValueError as err:
            logger.error(err)
            return {"error": "Invalid play ID format"}, 400

        return {"error": "Play not found"}, 404


class PlayDetailsResource(Resource):
    def __init__(self):
        super().__init__()

    def get(self, game_id: str, event_id: str):
        try:
            play = dataset_manager.get_play_details(game_id, event_id)
            if play is not None:
                return jsonify(play)
        except ValueError as err:
            logger.error(err)
            return {"error": "Invalid play ID format"}, 400

        return {"error": "Play not found"}, 404

    def post(self, game_id: str, event_id: str):
        update_play_data = request.get_json()
        update_play_data["event_id"] = event_id
        update_play_data["game_id"] = game_id

        try:
            df_update = pl.DataFrame(update_play_data, schema=UPDATE_PLAY_SCHEMA)
        except ValueError as err:
            logger.error(err)
            return {"error": "Invalid play ID format"}, 400

        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates.parquet")
        user_updates = user_updates.update(df_update, on=["game_id", "event_id"], how="full")
        user_updates.write_parquet(f"{DATA_DIR}/user_updates.parquet")

        return {"message": "Play updated successfully"}, 200


if __name__ == "__main__":
    """Use for debugging"""

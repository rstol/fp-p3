from flask import Response, jsonify
from flask_restful import Resource

from backend.resources.dataset_manager import DatasetManager
from backend.settings import TRACKING_DIR

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
            return {"error": "Play not found"}, 404
        except ValueError:
            return {"error": "Invalid play ID format"}, 400


class PlayDetailsResource(Resource):
    def get(self, game_id, event_id):
        try:
            video = dataset_manager.get_play_video(game_id, event_id)
            if video:
                return Response(video, mimetype="application/octet-stream")
            return {"error": "Play not found"}, 404
        except ValueError:
            return {"error": "Invalid play ID format"}, 400

import server.resources as res
from flask_restful import Api

API = "/api/v1/"


def add_routes(app):
    api = Api(app)

    # dummy
    api.add_resource(res.scatter_data.DatasetResource, API + "data/<string:name>")

    # NBA Data endpoints
    api.add_resource(res.nba_data.TeamsResource, API + "teams")
    api.add_resource(res.nba_data.TeamDetailsResource, API + "teams/<string:team_id>")
    api.add_resource(res.nba_data.TeamGamesResource, API + "teams/<string:team_id>/games")
    api.add_resource(res.nba_data.GamesResource, API + "games")
    api.add_resource(res.nba_data.GameDetailsResource, API + "games/<string:game_id>")
    api.add_resource(res.nba_data.GamePlaysResource, API + "games/<string:game_id>/plays")
    api.add_resource(
        res.nba_data.PlayDetailsResource, API + "games/<string:game_id>/plays/<string:play_id>"
    )

    return api

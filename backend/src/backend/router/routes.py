from flask_restful import Api

import backend.resources as res
from backend.settings import API


def add_routes(app):
    api = Api(app)

    # Scatter plot data for team plays
    api.add_resource(
        res.scatter_data.TeamPlaysScatterResource, API + "teams/<string:team_id>/plays/scatter"
    )
    api.add_resource(
        res.scatter_data.ClusterResource, API + "teams/<string:team_id>/cluster/<string:cluster_id>"
    )
    api.add_resource(
        res.scatter_data.ScatterPointResource,
        API + "teams/<string:team_id>/scatterpoint/<string:game_id>/<string:event_id>",
    )
    api.add_resource(
        res.scatter_data.BatchScatterPointResource, API + "teams/<string:team_id>/scatterpoints"
    )
    # NBA Data endpoints
    api.add_resource(res.nba_data.TeamsResource, API + "teams")
    api.add_resource(res.nba_data.TeamDetailsResource, API + "teams/<string:team_id>")
    api.add_resource(res.nba_data.TeamGamesResource, API + "teams/<string:team_id>/games")
    api.add_resource(res.nba_data.GamesResource, API + "games")
    api.add_resource(res.nba_data.GameDetailsResource, API + "games/<string:game_id>")
    api.add_resource(res.nba_data.GamePlaysResource, API + "games/<string:game_id>/plays")
    api.add_resource(
        res.nba_data.PlayRawDataResource, API + "games/<string:game_id>/plays/<string:play_id>/raw"
    )
    api.add_resource(
        res.nba_data.PlayVideoResource, API + "plays/<string:game_id>/<string:event_id>/video"
    )
    api.add_resource(
        res.nba_data.PlayDetailsResource, API + "plays/<string:game_id>/<string:event_id>"
    )

    return api

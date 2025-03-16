from flask_restful import Api
import server.resources as res

API = "/api/v1/"  # optional string


def add_routes(app):
    api = Api(app)

    api.add_resource(res.scatter_data.DatasetResource, API + "data/<string:name>")

    return api

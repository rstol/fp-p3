import os

from flask import Flask
from flask_cors import CORS

from backend.router.routes import add_routes


def create_app():
    app = Flask(__name__)  # static_url_path, static_folder, template_folder...
    CORS(app, resources={r"/*": {"origins": "*"}})
    add_routes(app)

    @app.route("/version")
    def version():
        return f"Job ID: {os.environ['JOB_ID']}\nCommit ID: {os.environ['COMMIT_ID']}"

    return app


app = create_app()

if __name__ == "__main__":
    # Only called in development mode as direct python execution
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

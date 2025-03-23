import json


class DatasetManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.teams = self._load_json("teams.json")
        self.games = self._load_json("games.json")

        self._play_cache = {}

    def _load_json(self, filename):
        with open(f"{self.data_dir}/{filename}", "r") as f:
            return json.load(f)

    def get_teams(self):
        return self.teams

    def get_team_details(self, team_id):
        for team in self.teams:
            if str(team["team_id"]) == str(team_id):
                return team
        return None

    def get_games_for_team(self, team_id):
        game_ids = [
            gid
            for gid, game in self.games.items()
            if str(team_id) in [str(game["home_team_id"]), str(game["visitor_team_id"])]
        ]
        return [self.games[gid] for gid in game_ids]

    def get_game_details(self, game_id):
        if game_id in self.games:
            return self.games[game_id]
        return None

    def get_plays_for_game(self, game_id):
        return self._load_game_plays(game_id)

    def get_play_id(self, game_id, play_id):
        plays = self._load_game_plays(game_id)
        for play in plays:
            if str(play["eventid"]) == str(play_id):
                return play
        return None

    def _load_game_plays(self, game_id):
        if game_id in self._play_cache:
            return self._play_cache[game_id]

        try:
            with open(f"{self.data_dir}/plays/{game_id}.json", "r") as f:
                plays = json.load(f)
                self._play_cache[game_id] = plays
                return plays
        except FileNotFoundError:
            return []

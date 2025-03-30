import json


class DatasetManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        # TODO load lines
        self.teams = self._load_jsonl("teams.jsonl")
        self.games = self._load_jsonl("games.jsonl")

    def _load_jsonl(self, filename):
        results = []
        with open(f"{self.data_dir}/{filename}", "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    results.append(json.loads(line))
        return results

    def _load_json(self, filename):
        with open(f"{self.data_dir}/{filename}", "r") as f:
            return json.load(f)

    def get_teams(self):
        return self.teams

    def get_team_details(self, team_id):
        for team in self.teams:
            if str(team["teamid"]) == str(team_id):
                return team
        return None

    def get_games(self):
        return self.games
          
  
    def get_games_for_team(self, team_id):
        return [
            game
            for game in self.games
            if str(team_id) in [str(game["home_team_id"]), str(game["visitor_team_id"])]
        ]

    def get_game_details(self, game_id):
        for game in self.games:
            if str(game["game_id"]) == str(game_id):
                return game
        return None

    def get_plays_for_game(self, game_id):
        return self._load_game_plays(game_id)

    def get_play_id(self, game_id, play_id):
        plays = self._load_game_plays(game_id)
        for play in plays:
            if str(play["event_id"]) == str(play_id):
                return play
        return None

    def _load_game_plays(self, game_id):
        try:
            return self._load_jsonl(f"plays/{game_id}.jsonl")
        except FileNotFoundError:
            return []

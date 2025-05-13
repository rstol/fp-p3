from backend.video.Team import Team


class Player:
    """A class for keeping info about the players"""

    def __init__(self, player):
        self.team = Team(player["teamid"])
        self.id = player["playerid"]
        self.x = player["x"]
        self.y = player["y"]
        self.color = self.team.color

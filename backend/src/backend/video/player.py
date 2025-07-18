from backend.video.team import Team


class Player:
    """A class for keeping info about the players"""

    def __init__(self, player):
        self.team = Team(player["teamid"])
        self.id = player["playerid"]
        self.x = player["x"]
        self.y = player["y"]

class Ball:
    """A class for keeping info about the balls"""

    def __init__(self, ball):
        self.x = ball["x"]
        self.y = ball["y"]
        self.radius = ball["z"]
        self.color = "#ff8c00"  # Hardcoded orange

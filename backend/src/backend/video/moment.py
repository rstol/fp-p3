from backend.video.Ball import Ball
from backend.video.Player import Player


class Moment:
    """A class for keeping info about the moments"""

    def __init__(self, moment):
        self.quarter = moment["quarter"]  # Hardcoded position for quarter in json
        self.game_clock = moment["game_clock"]  # Hardcoded position for game_clock in json
        self.shot_clock = moment["shot_clock"]  # Hardcoded position for shot_clock in json
        ball = moment["ball_coordinates"]  # Hardcoded position for ball in json
        self.ball = Ball(ball)
        players = moment["player_coordinates"]  # Hardcoded position for players in json

        self.players = [Player(player) for player in players]

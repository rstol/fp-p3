from backend.settings import COURT_LENGTH, COURT_WIDTH


class Constant:
    """A class for handling constants"""

    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = 0
    X_MAX = COURT_LENGTH
    Y_MIN = 0
    Y_MAX = COURT_WIDTH
    COL_WIDTH = 0.1
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    Y_CENTER = Y_MAX - DIFF / 1.5 - 3.2
    MESSAGE = "You can rerun the script and choose any event from 0 to "

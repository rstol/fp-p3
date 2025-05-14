import os
import tempfile
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

from backend.settings import RAW_DATA_HZ, SAMPLING_RATE, VIDEO_DATA_DIR
from backend.video.Constant import Constant
from backend.video.Moment import Moment

FPS = RAW_DATA_HZ // SAMPLING_RATE

# Set matplotlib to use a non-interactive backend
matplotlib.use("Agg")


class Event:
    """A class for handling and showing events"""

    # TODO: maybe make Event class that can basically only do a lookup if there is a video already
    #   and if there isn't, it has an initialize() function where it initializes itself (by getting its event json with the data manager)
    #   and a generate video function and then the video can be loaded afterwards.
    def __init__(self, event, home, visitor):
        moments = event["moments"]
        self.moments = [Moment(moment) for moment in moments]
        self.game_id = event["game_id"]
        self.event_id = event["event_id"]
        home_players = home["players"]
        guest_players = visitor["players"]
        players = home_players + guest_players
        player_ids = [player["playerid"] for player in players]
        player_names = [
            " ".join([player["firstname"], player["lastname"]]) for player in players
        ]
        player_jerseys = [player["jersey"] for player in players]
        values = list(zip(player_names, player_jerseys, strict=False))
        # Example: 101108: ['Chris Paul', '3']
        self.player_ids_dict = dict(zip(player_ids, values, strict=False))

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            if len(moment.players) != len(player_circles):
                continue  # skip
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)
            shot_clock = moment.shot_clock if moment.shot_clock is not None else 0.0
            clock_test = f"Quarter {moment.quarter:d}\n {int(moment.game_clock) % 3600 // 60:02d}:{int(moment.game_clock) % 60:02d}\n {shot_clock:03.1f}"
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle

    def generate_anim(self):
        # Leave some space for inbound passes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(Constant.X_MIN, Constant.X_MAX)
        ax.set_ylim(Constant.Y_MIN, Constant.Y_MAX)
        ax.axis("off")
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        for moment in self.moments:  # Try to find moment with 10 players
            if len(moment.players) == 10:
                start_moment = moment
                break

        player_dict = self.player_ids_dict
        players_missing = [p for p in start_moment.players if p.id not in player_dict]
        for p in players_missing:  # Handle missing player
            self.player_ids_dict.setdefault(p.id, ("Unkown", "N/A"))

        clock_info = ax.annotate(
            "",
            xy=[Constant.X_CENTER, Constant.Y_CENTER],
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

        annotations = [
            ax.annotate(
                player_dict[player.id][1],
                xy=[0, 0],
                color="w",
                horizontalalignment="center",
                verticalalignment="center",
                fontweight="bold",
            )
            for player in start_moment.players
        ]

        # Prepare table
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)

        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]

        home_players = [
            " #".join([player_dict[player.id][0], player_dict[player.id][1]])
            for player in sorted_players[:5]
        ]
        guest_players = [
            " #".join([player_dict[player.id][0], player_dict[player.id][1]])
            for player in sorted_players[5:]
        ]
        players_data = list(zip(home_players, guest_players, strict=False))

        table = plt.table(
            cellText=players_data,
            colLabels=column_labels,
            colColours=column_colours,
            colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
            loc="bottom",
            cellColours=cell_colours,
            fontsize=Constant.FONTSIZE,
            cellLoc="center",
        )
        table.scale(1, Constant.SCALE)
        # table_cells = table.properties()['child_artists']
        for key, cell in table.get_celld().items():
            cell.get_text().set_color("white")

        player_circles = [
            plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
            for player in start_moment.players
        ]
        ball_circle = plt.Circle(
            (0, 0), Constant.PLAYER_CIRCLE_SIZE, color=start_moment.ball.color
        )
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig,
            self.update_radius,
            fargs=(player_circles, ball_circle, annotations, clock_info),
            frames=len(self.moments),
            interval=Constant.INTERVAL,
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        court_path = os.path.join(script_dir, "court.png")
        court = plt.imread(court_path)
        plt.imshow(
            court,
            zorder=0,
            extent=[
                Constant.X_MIN,
                Constant.X_MAX - Constant.DIFF,
                Constant.Y_MAX,
                Constant.Y_MIN,
            ],
        )

        return fig, anim

    @lru_cache(maxsize=20)
    def generate_mp4(self, fps=FPS, bitrate=-1) -> bytes:
        """
        Generates and returns the raw binary data of the play animation as MP4.

        Parameters
        ----------
        self : Event
        fps : int, optional
            Frames per second for the video.
        bitrate : int, optional
            Bitrate of the output video in kbps.

        Returns
        -------
        bytes
            Raw binary MP4 data that can be sent directly in a Flask response.
        """
        fig, anim = self.generate_anim()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_filename = temp_file.name

        writer = FFMpegWriter(fps=fps, bitrate=bitrate, codec="h264")
        anim.save(temp_filename, writer=writer)

        plt.close(fig)

        with open(temp_filename, "rb") as f:
            video_data = f.read()

        os.unlink(temp_filename)

        return video_data

    def prerender(self, fps=FPS, bitrate=-1):
        fig, anim = self.generate_anim()

        filename = os.path.join(VIDEO_DATA_DIR, f"{self.game_id}_{self.event_id}.mp4")

        writer = FFMpegWriter(fps=fps, bitrate=bitrate, codec="h264")
        anim.save(filename, writer=writer)

        plt.close(fig)

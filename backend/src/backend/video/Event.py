from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

from backend.video.Constant import Constant
from backend.video.Moment import Moment


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
        player_names = [" ".join([player["firstname"], player["lastname"]]) for player in players]
        player_jerseys = [player["jersey"] for player in players]
        values = list(zip(player_names, player_jerseys, strict=False))
        # Example: 101108: ['Chris Paul', '3']
        self.player_ids_dict = dict(zip(player_ids, values, strict=False))

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)
            clock_test = f"Quarter {moment.quarter:d}\n {int(moment.game_clock) % 3600 // 60:02d}:{int(moment.game_clock) % 60:02d}\n {moment.shot_clock:03.1f}"
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle

    def generate_anim(self):
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN, Constant.X_MAX), ylim=(Constant.Y_MIN, Constant.Y_MAX))
        ax.axis("off")
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict

        clock_info = ax.annotate(
            "",
            xy=[Constant.X_CENTER, Constant.Y_CENTER],
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

        annotations = [
            ax.annotate(
                self.player_ids_dict[player.id][1],
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
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=start_moment.ball.color)
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
        court = plt.imread("court.png")
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

    def generate_jshtml(self):
        fig, anim = self.generate_anim()
        plt.close(fig)
        js_html = anim.to_jshtml(fps=1)
        return js_html

    def generate_mp4html(self, output=True, save=False, path=None):
        """
        Generates an HTML object with the video of the play.

        Warning: this function requires the ffmpeg package.
        Install with \n
        ``conda install ffmpeg`` \n
        Display output with \n
        ``from IPython.display import HTML`` \n
        ``HTML(<output>)``

        Parameters
        ----------
        self : Event
        output : Boolean, optional
            Whether or not to return an html object with the video.
        save : Boolean, optional
            Whether or not to save the video as an mp4.
            The file name is "game<game_id>play<play_id>.mp4".
        path : str, optional
            Path to the directory to save the video in.
            Default is to create an ouput directory and save it there.

        Returns
        -------
        str | None
            The html video or None if ``output`` was set to ``False``.

        Raises
        ------
        FileExistsError
            If the specified path already exists and is not a folder.
        """

        # 1) build the animation
        fig, anim = self.generate_anim()

        # 2) if requested, save out an .mp4
        if save:
            # determine output filepath
            if path is not None:
                outdir = Path(path)
            else:
                outdir = Path("output")

            outdir.mkdir(parents=True, exist_ok=True)
            filepath = outdir / f"game{self.game_id}play{self.event_id}.mp4"

            # use FFMpegWriter (requires ffmpeg installed)
            # TODO: What would the right fps be to get lifelike speed?
            writer = FFMpegWriter(fps=12)
            anim.save(str(filepath), writer=writer)
            print(f"Saved animation to {filepath}")

        # 3) close the figure so notebook/backend won’t display the static fallback
        plt.close(fig)

        # 4) return the HTML5 snippet for embedding
        if output:
            return anim.to_html5_video()

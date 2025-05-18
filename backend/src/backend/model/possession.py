import numpy as np

from backend.model.utils import get_game_time


class Agent:
    def __init__(self, teamid, agentid):
        self.teamid: int = teamid
        self.agentid: int = agentid
        self.x: list[float] = []
        self.y: list[float] = []
        self.game_times = []
        self.v = []
        self.valid = []


class Possession:
    def __init__(self, event, gameid):
        self.gameid = gameid
        self.eventid = event["event_info"]["id"]
        self.moments = event["moments"]
        teams = {event["home"]["teamid"], event["visitor"]["teamid"]}
        self.off_teamid = int(event["event_info"]["possession_team_id"])
        self.def_teamid = int((teams - {self.off_teamid}).pop())
        self.game_time_list, self.shot_clock_list, self.game_clock_list = self.get_time()

        self.agents = self.get_agents_list()

    def get_time(self):
        shot_clock_list = []
        game_time_list = []
        game_clock_list = []
        for moment in self.moments:
            game_clock_list.append(moment["game_clock"])
            shot_clock_list.append(moment["shot_clock"])
            game_time_list.append(get_game_time(moment["game_clock"], moment["quarter"]))
        return game_time_list, shot_clock_list, game_clock_list

    def get_agents_list(self):
        player_data = self.moments[0]["player_coordinates"]
        agents = []
        for agent in player_data:
            agents.append(Agent(int(agent["teamid"]), int(agent["playerid"])))
        agents.append(Agent(-1, -1))  # Ball is an Agent
        agents = self.get_agents_feature(agents)
        return agents

    def velocity(self, traj):
        dif = traj[1:] - traj[:-1]
        v = np.sqrt((dif**2).sum(1)) / 0.04  # feet / second
        return v

    def get_agents_feature(self, agents_list: list[Agent]):
        for agent in agents_list:
            agent.game_times = self.game_time_list
            # get x，y
            for moment in self.moments:
                valid = 0
                ball_coord = moment["ball_coordinates"]
                if agent.agentid == -1:
                    agent.x.append(ball_coord["x"])
                    agent.y.append(ball_coord["y"])
                    valid = 1
                    agent.valid.append(valid)
                for player_coord in moment["player_coordinates"]:
                    if agent.agentid == player_coord["playerid"]:
                        agent.x.append(player_coord["x"])
                        agent.y.append(player_coord["y"])
                        valid = 1
                        agent.valid.append(valid)
                        continue
                if valid == 0:  # invalid agent coordinates
                    agent.x.append(-1)
                    agent.y.append(-1)
                    agent.valid.append(0)

            # get velocities
            coords = np.stack((agent.x, agent.y), axis=1)
            if len(coords) < 2:
                continue
            v = self.velocity(coords)
            agent.v = np.append(v, v[-1])

        return agents_list

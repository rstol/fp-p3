export interface DataPoint {
  X1: number;
  X2: number;
  cluster: string;
}

export type DataArray = DataPoint[];

export type Player = {
  firstname: string;
  lastname: string;
  jersey: string;
  playerid: number;
  position: string;
};

export type Team = {
  abbreviation: string;
  name: string;
  teamid: number;
  players: Player[];
};

export type Game = {
  game_id: string;
  game_date: string; // or use `Date` if you plan to parse it
  home_team_id: number;
  visitor_team_id: number;
};

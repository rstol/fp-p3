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

export type Point = {
  x: number;
  y: number;
  cluster: number;
  event_id: string;
  play_type?: string;
  description?: string;
  period?: number;
  game_id: string;
};

export type Play = Pick<
  Point,
  'game_id' | 'event_id' | 'cluster' | 'play_type' | 'description' | 'period'
>;

export type DetailedPlay = {
  game_id: string;
  event_id: string;
  event_type?: number;
  event_desc_home?: string;
  event_desc_away?: string;
  period?: number;
  time_remaining?: string;
  player1_id?: number;
  player1_name?: string;
  player2_id?: number;
  player2_name?: string;
};

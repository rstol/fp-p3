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
  game_date: string;
  home_team_id: number;
  visitor_team_id: number;
};

export interface PlayerInfo {
  player_id: number;
  team: string;
  team_id: number;
  player_name?: string;
}

export interface PlayDetail {
  game_id: string;
  event_id: string;

  primary_player_info?: PlayerInfo | null;
  secondary_player_info?: PlayerInfo | null;

  event_type: number;
  possession_team_id?: number | null;

  event_desc_home: string | null;
  event_desc_away: string | null;

  game_date?: string;
  event_score?: string;
  quarter?: number;
}

export interface Point {
  x: number;
  y: number;
  event_id: string;
  game_id: string;
  event_desc_home: string | null;
  event_desc_away: string | null;
  game_date?: string;
  event_type: number;
  is_tagged: boolean;
  note: null | string;
  score: string | number;
  similarity_distance: number;
}

export interface ClusterMetadata {
  cluster_id: string;
  cluster_label?: string;
}

export interface ClusterData extends ClusterMetadata {
  points: Array<Point>;
}

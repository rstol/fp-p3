export const BASE_URL =
  process.env.NODE_ENV === 'production'
    ? `http://be.${window.location.hostname}/api/v1`
    : 'http://localhost:8080/api/v1';

export enum GameFilter {
  LAST1 = 1,
  LAST2 = 2,
  LAST3 = 3,
  LAST4 = 4,
  LAST5 = 5,
}
export const TeamIDs = [1610612748, 1610612752, 1610612755];
export const EventType: Record<number, string> = {
  1: 'made',
  2: 'miss',
  3: 'free_throw',
  4: 'rebound',
  5: 'turnover',
  6: 'foul',
  7: 'violation',
  8: 'substitution',
  9: 'timeout',
  10: 'jump_ball',
  11: 'ejection',
  12: 'start_period',
  13: 'end_period',
};

export enum PlayActions {
  UpdateAllPlayFields = 'UpdateAllPlayFields',
  UpdatePlayNote = 'UpdatePlayNote',
  UpdatePlayCluster = 'UpdatePlayCluster',
}

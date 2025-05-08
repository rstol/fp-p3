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

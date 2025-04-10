export const BASE_URL =
  process.env.NODE_ENV === 'production'
    ? `http://be.${window.location.hostname}/api/v1`
    : 'http://localhost:8080/api/v1';

export enum GameFilter {
  LAST3 = 'last_3',
  LAST4 = 'last_4',
  LAST5 = 'last_5',
}

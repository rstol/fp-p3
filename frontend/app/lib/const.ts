export const BASE_URL =
  process.env.NODE_ENV === 'production'
    ? `http://be.${window.location.hostname}/api/v1`
    : 'http://localhost:8080/api/v1';

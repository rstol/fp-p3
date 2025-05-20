import localforage from 'localforage';
import { BASE_URL } from './const';

function createCacheKey(fullUrl: string, includeSearch: boolean = false): string {
  const urlObj = new URL(fullUrl);
  return includeSearch ? urlObj.pathname + urlObj.search : urlObj.pathname;
}

const CURRENT_COMMIT_HASH = __COMMIT_HASH__; // injected at build time
const COMMIT_KEY = 'app_commit_hash';

export const purgeCacheOnGitCommitChange = async () => {
  const storedHash = localStorage.getItem(COMMIT_KEY);

  if (storedHash !== CURRENT_COMMIT_HASH) {
    console.log('Commit hash changed. Purging localForage cache...');
    await localforage.clear();
    localStorage.setItem(COMMIT_KEY, CURRENT_COMMIT_HASH);
  } else {
    console.log('Commit hash unchanged. Cache is valid.');
  }
};

export const purgeScatterDataCache = async (teamID: string | null) => {
  if (!teamID) return;
  const keys = await localforage.keys();
  keys.forEach((key) => {
    if (key.startsWith(`${BASE_URL}/teams/${teamID}/plays/scatter`))
      localforage
        .removeItem(key)
        .then()
        .catch((err) => console.log(err));
  });
};

export async function fetchWithCache<T>(url: string, includeSearch: boolean = false): Promise<T> {
  const key = createCacheKey(url, includeSearch);
  const cached = await localforage.getItem<T>(key);
  if (cached) {
    console.log(`Cache hit for ${key}`);
    return cached;
  }
  console.log(`Cache miss for ${key}`);

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch from ${url}`);

  const data: T = await res.json();
  await localforage.setItem(key, data);
  return data;
}

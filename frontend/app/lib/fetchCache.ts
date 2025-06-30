import { useDashboardStore } from './stateStore';

const CURRENT_COMMIT_HASH = __COMMIT_HASH__; // injected at build time
const COMMIT_KEY = 'app_commit_hash';

export const purgeCacheOnGitCommitChange = async () => {
  const storedHash = localStorage.getItem(COMMIT_KEY);

  if (storedHash !== CURRENT_COMMIT_HASH) {
    console.log('Commit hash changed. Purging localForage cache...');
    useDashboardStore.persist.clearStorage();
    localStorage.setItem(COMMIT_KEY, CURRENT_COMMIT_HASH);
  } else {
    console.log('Commit hash unchanged. Cache is valid.');
  }
};

export async function fetchWrapper<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch from ${url}`, { cause: res.statusText });

  return res.json() as Promise<T>;
}

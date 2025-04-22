/**
 * Client-side tag management for basketball plays
 * 
 * This module handles storing and retrieving play tags in the browser's local storage
 * rather than sending them to the server. This approach:
 * 
 * 1. Keeps tags isolated per user
 * 2. Avoids server-side storage and concurrency issues
 * 3. Makes the application work without a persistent backend
 */

// Unique key for storing tags in localStorage
const TAGS_STORAGE_KEY = 'basketball-play-tags';

// Interface for the tag storage schema
interface TagStore {
  version: number;
  tags: Record<string, string>;
  lastUpdated: number;
}

/**
 * Create a key for a play that can be used for tag storage
 * @param gameId The game ID
 * @param playId The play/event ID
 * @returns A unique string key
 */
export function createTagKey(gameId: string, playId: string): string {
  return `${gameId}:${playId}`;
}

/**
 * Initialize the tag store in localStorage if it doesn't exist
 */
function initializeTagStore(): TagStore {
  const defaultStore: TagStore = {
    version: 1,
    tags: {},
    lastUpdated: Date.now()
  };
  
  try {
    const existingData = localStorage.getItem(TAGS_STORAGE_KEY);
    if (existingData) {
      return JSON.parse(existingData);
    }
  } catch (error) {
    console.error('Error loading tags from localStorage', error);
  }
  
  // Initialize empty store
  localStorage.setItem(TAGS_STORAGE_KEY, JSON.stringify(defaultStore));
  return defaultStore;
}

/**
 * Get all stored tags
 * @returns The complete tag store object
 */
export function getAllTags(): TagStore {
  return initializeTagStore();
}

/**
 * Get the tag for a specific play
 * @param gameId The game ID
 * @param playId The play/event ID 
 * @returns The tag value or null if not found
 */
export function getTag(gameId: string, playId: string): string | null {
  const store = initializeTagStore();
  const key = createTagKey(gameId, playId);
  return store.tags[key] || null;
}

/**
 * Set the tag for a specific play
 * @param gameId The game ID 
 * @param playId The play/event ID
 * @param tag The tag value
 */
export function setTag(gameId: string, playId: string, tag: string): void {
  const store = initializeTagStore();
  const key = createTagKey(gameId, playId);
  
  // Update tag
  store.tags[key] = tag;
  store.lastUpdated = Date.now();
  
  // Save updated store
  localStorage.setItem(TAGS_STORAGE_KEY, JSON.stringify(store));
}

/**
 * Export all tags to a JSON string for backup or sharing
 * @returns JSON string containing all tag data
 */
export function exportTags(): string {
  const store = initializeTagStore();
  return JSON.stringify(store);
}

/**
 * Import tags from a JSON string
 * @param jsonData JSON string to import
 * @param merge Whether to merge with existing tags (true) or replace them (false)
 * @returns Boolean indicating success
 */
export function importTags(jsonData: string, merge = false): boolean {
  try {
    const importedStore = JSON.parse(jsonData) as TagStore;
    
    // Validate the imported data
    if (!importedStore || !importedStore.tags || typeof importedStore.tags !== 'object') {
      return false;
    }
    
    if (merge) {
      // Merge with existing tags
      const currentStore = initializeTagStore();
      importedStore.tags = { ...currentStore.tags, ...importedStore.tags };
    }
    
    // Save the imported store
    localStorage.setItem(TAGS_STORAGE_KEY, JSON.stringify(importedStore));
    return true;
  } catch (error) {
    console.error('Error importing tags', error);
    return false;
  }
}

/**
 * Apply tags to a collection of play data
 * This adds a "tag" property to each point using stored client-side tags
 * @param points Array of play data points
 * @returns The same array with tags applied
 */
export function applyTagsToData<T extends { game_id: string, event_id: string, cluster: number }>(
  points: T[]
): (T & { tag: string })[] {
  const store = initializeTagStore();
  
  return points.map(point => {
    const key = createTagKey(point.game_id, point.event_id);
    // Use stored tag if available, otherwise use cluster number
    const tag = store.tags[key] || String(point.cluster);
    return { ...point, tag };
  });
}

/**
 * Reset all stored tags
 */
export function resetTags(): void {
  localStorage.removeItem(TAGS_STORAGE_KEY);
  initializeTagStore();
}

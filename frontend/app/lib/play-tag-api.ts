/**
 * API utilities for working with play tags
 */

interface TagResponse {
  tags: string[];
}

/**
 * Gets tags for a specific play
 */
export async function getPlayTags(gameId: string, playId: string): Promise<string[]> {
  try {
    const response = await fetch(
      `/api/v1/games/${gameId}/plays/${playId}/tags`
    );
    
    if (!response.ok) {
      console.error('Failed to fetch tags:', response.statusText);
      return [];
    }
    
    const data = await response.json() as TagResponse;
    return data.tags || [];
  } catch (error) {
    console.error('Error fetching tags:', error);
    return [];
  }
}

/**
 * Adds a tag to a play
 */
export async function addPlayTag(gameId: string, playId: string, tag: string): Promise<string[]> {
  try {
    const response = await fetch(
      `/api/v1/games/${gameId}/plays/${playId}/tags`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tag: tag.trim() }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error saving tag: ${response.statusText}`);
    }
    
    const data = await response.json() as TagResponse;
    return data.tags || [];
  } catch (error) {
    console.error('Error saving tag:', error);
    return [];
  }
}

/**
 * Removes a tag from a play
 */
export async function removePlayTag(gameId: string, playId: string, tag: string): Promise<string[]> {
  try {
    const response = await fetch(
      `/api/v1/games/${gameId}/plays/${playId}/tags`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tag,
          operation: 'remove'
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error removing tag: ${response.statusText}`);
    }
    
    const data = await response.json() as TagResponse;
    return data.tags || [];
  } catch (error) {
    console.error('Error removing tag:', error);
    return [];
  }
}

/**
 * Updates all tags for a play
 */
export async function updatePlayTags(gameId: string, playId: string, tags: string[]): Promise<string[]> {
  try {
    const response = await fetch(
      `/api/v1/games/${gameId}/plays/${playId}/tags`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tags,
          operation: 'update'
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`Error updating tags: ${response.statusText}`);
    }
    
    const data = await response.json() as TagResponse;
    return data.tags || [];
  } catch (error) {
    console.error('Error updating tags:', error);
    return [];
  }
}

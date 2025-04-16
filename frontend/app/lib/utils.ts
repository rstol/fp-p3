import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { Play } from '~/types/data';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getPlayId(play: Play | null): string {
  if (!play) return '';
  return `${play.game_id}-${play.event_id}`;
}

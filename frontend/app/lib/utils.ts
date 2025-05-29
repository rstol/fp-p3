import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { Point } from '~/types/data';
import { v4 as uuidv4 } from 'uuid';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getPointId(point: Point | null): string {
  if (!point) return '';
  return `${point.game_id}-${point.event_id}`;
}

export const generateTagId = () => {
  return uuidv4();
};

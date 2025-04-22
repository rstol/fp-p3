import { create } from 'zustand';
import type { Play } from '~/types/data';

interface DashboardState {
  selectedPlay: Play | null;
  tagUpdateCounter: number; // Track tag updates to trigger re-renders
  updatePlay: (play: Play) => void;
  triggerTagUpdate: () => void; // New method to signal tag updates
}

export const useDashboardStore = create<DashboardState>((set) => ({
  selectedPlay: null,
  tagUpdateCounter: 0,
  updatePlay: (play) => set({ selectedPlay: play }),
  triggerTagUpdate: () => set((state) => ({ tagUpdateCounter: state.tagUpdateCounter + 1 })),
}));

import { create } from 'zustand';
import type { Play } from '~/types/data';

type State = {
  selectedPlay: Play | null;
};
type Action = {
  updatePlay: ({}: Play) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  selectedPlay: null,
  updatePlay: (selectedPlay) => set(() => ({ selectedPlay })),
  resetPlay: () => set(() => ({ selectedPlay: null })),
}));

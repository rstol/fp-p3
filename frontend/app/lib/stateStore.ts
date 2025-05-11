import { create } from 'zustand';
import type { Play } from '~/types/data';

type State = {
  selectedPlay: Play | null;
};
type Action = {
  updatePlay: (play: Play) => void;
  resetPlay: () => void;
  updateSelectedPlayCluster: (clusterId: number) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  selectedPlay: null,
  updatePlay: (selectedPlay) => set(() => ({ selectedPlay })),
  resetPlay: () => set(() => ({ selectedPlay: null })),
  updateSelectedPlayCluster: (clusterId) =>
    set((state) => {
      if (state.selectedPlay) {
        return {
          selectedPlay: { ...state.selectedPlay, cluster: clusterId },
        };
      }
      return state;
    }),
}));

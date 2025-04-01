import { create } from 'zustand';

type State = {
  homeTeamId: null | string;
};
type Action = {
  updateHomeTeamId: (newHomeTeamId: string) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  homeTeamId: null,
  updateHomeTeamId: (newHomeTeamId) => set({ homeTeamId: newHomeTeamId }),
}));

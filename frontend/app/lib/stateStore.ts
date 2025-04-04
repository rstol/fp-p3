import { create } from 'zustand';

export enum GameFilter {
  LAST3 = 'last_3',
  LAST4 = 'last_4',
  LAST5 = 'last_5',
}

type State = {
  homeTeamId: null | string;
  gameFilter: GameFilter;
};
type Action = {
  updateHomeTeamId: (newHomeTeamId: string) => void;
  updateGameFilter: (newGameFilter: GameFilter) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  homeTeamId: null,
  gameFilter: GameFilter.LAST3,
  updateGameFilter: (newGameFilter) => set({ gameFilter: newGameFilter }),
  updateHomeTeamId: (newHomeTeamId) => set({ homeTeamId: newHomeTeamId }),
}));

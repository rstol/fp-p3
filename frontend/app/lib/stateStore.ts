import { create } from 'zustand';
import type { Play } from '~/types/data';

// Helper to generate a unique ID for a play
const getPlayId = (play: Play): string => `${play.game_id}-${play.event_id}`;

type State = {
  selectedPlay: Play | null;
  pendingClusterUpdates: Map<string, number>;
  stagedChangesCount: number;
  selectedTeamId: string | null;
};

type Action = {
  updatePlay: (play: Play) => void;
  resetPlay: () => void;
  stageSelectedPlayClusterUpdate: (clusterId: number) => void;
  clearPendingClusterUpdates: () => void;
  setSelectedTeamId: (teamId: string | null) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  selectedPlay: null,
  pendingClusterUpdates: new Map<string, number>(),
  stagedChangesCount: 0,
  selectedTeamId: null,
  updatePlay: (selectedPlay) => set(() => ({ selectedPlay })),
  resetPlay: () =>
    set(() => ({
      selectedPlay: null,
    })),
  stageSelectedPlayClusterUpdate: (clusterId) =>
    set((state) => {
      const { selectedPlay, pendingClusterUpdates } = state;
      if (selectedPlay) {
        const playId = getPlayId(selectedPlay);
        const newPendingUpdates = new Map(pendingClusterUpdates);
        newPendingUpdates.set(playId, clusterId);

        return {
          selectedPlay: { ...selectedPlay, cluster: clusterId },
          pendingClusterUpdates: newPendingUpdates,
          stagedChangesCount: newPendingUpdates.size,
        };
      }
      return state;
    }),
  clearPendingClusterUpdates: () =>
    set(() => ({
      pendingClusterUpdates: new Map<string, number>(),
      stagedChangesCount: 0,
    })),
  setSelectedTeamId: (teamId) => {
    set(() => ({ selectedTeamId: teamId }));
  },
}));

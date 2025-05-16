import { create } from 'zustand';
import type { Point, Play } from '~/types/data';

// Helper to generate a unique ID for a play
const getPlayId = (play: Play | Point): string => `${play.game_id}-${play.event_id}`;

type State = {
  selectedPoint: Point | null;
  pendingClusterUpdates: Map<string, number>;
  stagedChangesCount: number;
  selectedTeamId: string | null;
};

type Action = {
  updatePoint: (point: Point) => void;
  resetPoint: () => void;
  stageSelectedPlayClusterUpdate: (clusterId: number) => void;
  clearPendingClusterUpdates: () => void;
  setSelectedTeamId: (teamId: string | null) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  selectedPoint: null,
  pendingClusterUpdates: new Map<string, number>(),
  stagedChangesCount: 0,
  selectedTeamId: null,
  updatePoint: (selectedPoint) => set(() => ({ selectedPoint })),
  resetPoint: () =>
    set(() => ({
      selectedPoint: null,
    })),
  stageSelectedPlayClusterUpdate: (clusterId) =>
    set((state) => {
      const { selectedPoint, pendingClusterUpdates } = state;
      if (selectedPoint) {
        const playId = getPlayId(selectedPoint);
        const newPendingUpdates = new Map(pendingClusterUpdates);
        newPendingUpdates.set(playId, clusterId);

        return {
          selectedPoint: { ...selectedPoint, cluster: clusterId },
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

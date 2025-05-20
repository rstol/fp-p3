import { create } from 'zustand';
import type { Point } from '~/types/data';
import { getPointId } from './utils';

type State = {
  selectedPoint: Point | null;
  pendingClusterUpdates: Map<string, string>;
  stagedChangesCount: number;
  selectedTeamId: string | null;
};

type Action = {
  updatePoint: (point: Point) => void;
  resetPoint: () => void;
  stageSelectedPlayClusterUpdate: (clusterId: string) => void;
  clearPendingClusterUpdates: () => void;
  setSelectedTeamId: (teamId: string | null) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  selectedPoint: null,
  pendingClusterUpdates: new Map<string, string>(),
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
        const playId = getPointId(selectedPoint);
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
      pendingClusterUpdates: new Map<string, string>(),
      stagedChangesCount: 0,
    })),
  setSelectedTeamId: (teamId) => {
    set(() => ({ selectedTeamId: teamId }));
  },
}));

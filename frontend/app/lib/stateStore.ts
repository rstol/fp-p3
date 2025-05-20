import { create } from 'zustand';
import type { Point } from '~/types/data';

type State = {
  selectedPoint: Point | null;
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
  stagedChangesCount: 0,
  selectedTeamId: null,
  updatePoint: (selectedPoint) => set(() => ({ selectedPoint })),
  resetPoint: () =>
    set(() => ({
      selectedPoint: null,
    })),
  stageSelectedPlayClusterUpdate: (clusterId) =>
    set((state) => {
      const { selectedPoint, stagedChangesCount } = state;
      if (!selectedPoint) return state;

      // TODO only count if cluster assignment changed
      if (selectedPoint.cluster === clusterId) return state;

      return {
        selectedPoint: { ...selectedPoint, cluster: clusterId },
        stagedChangesCount: stagedChangesCount + 1,
      };
    }),
  clearPendingClusterUpdates: () =>
    set(() => ({
      stagedChangesCount: 0,
    })),
  setSelectedTeamId: (teamId) => {
    set(() => ({ selectedTeamId: teamId }));
  },
}));

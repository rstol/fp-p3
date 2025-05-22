import { create } from 'zustand';
import type { Point } from '~/types/data';

type State = {
  selectedPoint: Point | null;
  stagedChangesCount: number;
  selectedClusterId: string | null;
};

type Action = {
  updateSelectedPoint: (point: Point) => void;
  resetSelectedPoint: () => void;
  stageSelectedPlayClusterUpdate: (clusterId: string) => void;
  clearPendingClusterUpdates: () => void;
  updateSelectedClusterId: (clusterId: string) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  selectedPoint: null,
  stagedChangesCount: 0,
  selectedClusterId: null,
  updateSelectedPoint: (selectedPoint) => set(() => ({ selectedPoint })),
  updateSelectedClusterId: (selectedClusterId) => set(() => ({ selectedClusterId })),
  resetSelectedPoint: () =>
    set(() => ({
      selectedPoint: null,
    })),
  stageSelectedPlayClusterUpdate: (clusterId) =>
    set((state) => {
      const { selectedPoint, stagedChangesCount } = state;
      if (!selectedPoint) return state;

      // TODO only count if cluster assignment changed
      // if (selectedPoint.cluster === clusterId) return state;

      return {
        selectedPoint: { ...selectedPoint, cluster: clusterId },
        stagedChangesCount: stagedChangesCount + 1,
      };
    }),
  clearPendingClusterUpdates: () =>
    set(() => ({
      stagedChangesCount: 0,
    })),
}));

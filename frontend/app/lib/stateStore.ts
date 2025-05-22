import { create } from 'zustand';
import type { ClusterMetadata, Point } from '~/types/data';

type State = {
  selectedPoint: Point | null;
  stagedChangesCount: number;
  selectedCluster: ClusterMetadata | null;
};

type Action = {
  updateSelectedPoint: (point: Point) => void;
  resetSelectedPoint: () => void;
  stageSelectedPlayClusterUpdate: (clusterId: string) => void;
  clearPendingClusterUpdates: () => void;
  updateSelectedCluster: (cluster: ClusterMetadata) => void;
};

export const useDashboardStore = create<State & Action>((set) => ({
  selectedPoint: null,
  stagedChangesCount: 0,
  selectedCluster: null,
  updateSelectedPoint: (selectedPoint) => set(() => ({ selectedPoint })),
  updateSelectedCluster: ({ cluster_id, cluster_label }) =>
    set(() => ({
      selectedCluster: { cluster_id, cluster_label },
    })),
  resetSelectedPoint: () =>
    set(() => ({
      selectedPoint: null,
    })),
  stageSelectedPlayClusterUpdate: (clusterId) =>
    set((state) => {
      const { stagedChangesCount } = state;
      return {
        stagedChangesCount: stagedChangesCount + 1,
      };
    }),
  clearPendingClusterUpdates: () =>
    set(() => ({
      stagedChangesCount: 0,
    })),
}));

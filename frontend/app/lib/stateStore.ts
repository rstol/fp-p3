import { create } from 'zustand';
import type { ClusterData, ClusterMetadata, Point } from '~/types/data';
import { getPointId } from './utils';
import { persist, createJSONStorage } from 'zustand/middleware';

type State = {
  clusters: ClusterData[];
  selectedPoint: Point | null;
  stagedChangesCount: number;
  selectedCluster: ClusterMetadata | null;
};

type Action = {
  updateSelectedPoint: (point: Point | null) => void;
  resetSelectedPoint: () => void;
  stageSelectedPlayClusterUpdate: (clusterId: string, count?: number) => void;
  clearPendingClusterUpdates: () => void;
  clearSelectedCluster: () => void;
  updateSelectedCluster: (cluster: ClusterMetadata) => void;
  setClusters: (clusters: ClusterData[]) => void;
  updatePointNote: (point: Point, newNote: string) => void;
  movePointToCluster: (point: Point, targetClusterId: string) => void;
  createNewClusterWithPoint: (newCluster: ClusterMetadata, point: Point) => void;
  updateIsTagged: (point: Point, isTagged?: boolean) => void;
  updateClusterLabel: (clusterId: string, newLabel: string) => void;
};

type Store = State & Action;
export const useDashboardStore = create<Store>()(
  persist(
    (set) => ({
      clusters: [],
      selectedPoint: null,
      stagedChangesCount: 0,
      selectedCluster: null,
      setClusters: (clusters) => set(() => ({ clusters })),
      updateSelectedPoint: (selectedPoint) => set(() => ({ selectedPoint })),
      updateSelectedCluster: ({ cluster_id, cluster_label }) =>
        set(() => ({
          selectedCluster: { cluster_id, cluster_label },
        })),
      clearSelectedCluster: () =>
        set(() => ({
          selectedCluster: null,
        })),
      resetSelectedPoint: () =>
        set(() => ({
          selectedPoint: null,
        })),
      stageSelectedPlayClusterUpdate: (clusterId, count = 1) =>
        set((state) => {
          const { stagedChangesCount } = state;
          return {
            stagedChangesCount: stagedChangesCount + count,
          };
        }),
      clearPendingClusterUpdates: () =>
        set(() => ({
          stagedChangesCount: 0,
        })),
      updatePointNote: (point, newNote) =>
        set((state) => ({
          clusters: state.clusters.map(({ points, ...cluster }) => ({
            ...cluster,
            points: points.map((p) =>
              getPointId(p) === getPointId(point) ? { ...p, note: newNote } : p,
            ),
          })),
        })),
      updateClusterLabel: (clusterId, newLabel) =>
        set((state) => ({
          clusters: state.clusters.map((cluster) =>
            cluster.cluster_id === clusterId ? { ...cluster, cluster_label: newLabel } : cluster,
          ),
        })),
      movePointToCluster: (point, targetClusterId) =>
        set((state) => {
          let movedPoint: Point | null = null;

          const updatedClusters = state.clusters
            .map((cluster) => {
              const filteredPoints = cluster.points.filter((p) => {
                if (getPointId(p) === getPointId(point)) {
                  movedPoint = { ...p, is_tagged: true };
                  return false;
                }
                return true;
              });
              return { ...cluster, points: filteredPoints };
            })
            .map((cluster) =>
              cluster.cluster_id === targetClusterId && movedPoint
                ? { ...cluster, points: [...cluster.points, movedPoint] }
                : cluster,
            );

          return { clusters: updatedClusters };
        }),
      updateIsTagged: (point, isTagged = true) =>
        set((state) => ({
          clusters: state.clusters.map((cluster) => ({
            ...cluster,
            points: cluster.points.map((p) =>
              getPointId(p) === getPointId(point) ? { ...p, is_tagged: isTagged } : p,
            ),
          })),
        })),
      createNewClusterWithPoint: (newCluster, point) =>
        set((state) => {
          const updatedClusters = state.clusters.map((cluster) => ({
            ...cluster,
            points: cluster.points.filter((p) => getPointId(p) !== getPointId(point)),
          }));

          return {
            clusters: [
              ...updatedClusters,
              {
                ...newCluster,
                points: [{ ...point, is_tagged: true }],
              },
            ],
          };
        }),
    }),
    {
      name: 'useDashboardStore',
      storage: createJSONStorage(() => sessionStorage),
    },
  ),
);

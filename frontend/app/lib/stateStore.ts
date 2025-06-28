import { create } from 'zustand';
import type { Tag, ClusterData, ClusterMetadata, Point, Game, Team } from '~/types/data';
import { getPointId } from './utils';
import { persist, createJSONStorage } from 'zustand/middleware';

type State = {
  teams: Team[];
  games: Game[];
  clusters: ClusterData[];
  tags: Tag[];
  selectedPoint: Point | null;
  stagedChangesCount: number;
  selectedCluster: ClusterMetadata | null;
  playbackSpeed: number;
};

type Action = {
  setTeams: (teams: Team[]) => void;
  setGames: (games: Game[]) => void;
  updateSelectedPoint: (point: Point | null) => void;
  resetSelectedPoint: () => void;
  stageSelectedPlayClusterUpdate: (clusterId: string, count?: number) => void;
  clearPendingClusterUpdates: () => void;
  clearSelectedCluster: () => void;
  updateSelectedCluster: (cluster: ClusterMetadata) => void;
  setClusters: (clusters: ClusterData[]) => void;
  setPlaybackSpeed: (speed: number) => void;
  updatePointNote: (point: Point, newNote: string) => void;
  updatePointTags: (point: Point, newTags: Tag[]) => void;
  movePointToCluster: (point: Point, targetClusterId: string) => void;
  createNewClusterWithPoint: (newCluster: ClusterMetadata, point: Point) => void;
  updateManuallyClustered: (point: Point, manuallyClustered?: boolean) => void;
  updateClusterLabel: (clusterId: string, newLabel: string) => void;
};

const mergePointTags = (newPoint: Point, existingPoint?: Point) => ({
  ...newPoint,
  tags: existingPoint?.tags
    ? [
        ...(existingPoint.tags || []),
        ...(newPoint.tags || []).filter(
          (newTag) =>
            !existingPoint.tags?.some((existingTag) => existingTag.tag_id === newTag.tag_id),
        ),
      ]
    : newPoint.tags,
});

type Store = State & Action;
export const useDashboardStore = create<Store>()(
  persist(
    (set) => ({
      games: [],
      teams: [],
      clusters: [],
      tags: [],
      selectedPoint: null,
      stagedChangesCount: 0,
      selectedCluster: null,
      playbackSpeed: 1,
      setTeams: (teams) =>
        set(() => ({
          teams,
        })),
      setGames: (games) =>
        set(() => ({
          games,
        })),
      setClusters: (clusters) =>
        set((state) => ({
          clusters: clusters.map((cluster) => ({
            ...cluster,
            points: cluster.points.map((newPoint) => {
              const existingPoint = state.clusters
                .flatMap((c) => c.points)
                .find((p) => getPointId(p) === getPointId(newPoint));

              return mergePointTags(newPoint, existingPoint);
            }),
          })),
        })),
      updateSelectedPoint: (selectedPoint) => set(() => ({ selectedPoint })),
      updateSelectedCluster: ({ cluster_id, cluster_label }) =>
        set(() => ({
          selectedCluster: { cluster_id, cluster_label },
        })),
      setPlaybackSpeed: (speed: number) => set(() => ({ playbackSpeed: speed })),
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
      updatePointTags: (point, newTags) =>
        set((state) => {
          const existingTagIds = new Set((point.tags || []).map((tag) => tag.tag_id));
          const uniqueNewPointTags = newTags.filter((tag) => !existingTagIds.has(tag.tag_id));

          const globalTagIds = new Set(state.tags.map((tag) => tag.tag_id));
          const uniqueNewTags = uniqueNewPointTags.filter((tag) => !globalTagIds.has(tag.tag_id));

          return {
            tags: [...state.tags, ...uniqueNewTags],
            clusters: state.clusters.map(({ points, ...cluster }) => ({
              ...cluster,
              points: points.map((p) =>
                getPointId(p) === getPointId(point)
                  ? { ...p, tags: [...(p.tags || []), ...uniqueNewPointTags] }
                  : p,
              ),
            })),
          };
        }),

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
                  movedPoint = {
                    ...p,
                    original_cluster: p.original_cluster ?? cluster,
                    is_tagged: true,
                  };
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
      updateManuallyClustered: (point, manuallyClustered = true) =>
        set((state) => ({
          clusters: state.clusters.map((cluster) => ({
            ...cluster,
            points: cluster.points.map((p) =>
              getPointId(p) === getPointId(point) ? { ...p, is_tagged: manuallyClustered } : p,
            ),
          })),
        })),
      createNewClusterWithPoint: (newCluster, point) =>
        set((state) => {
          const oldCluster = state.clusters.find((cluster) => {
            cluster.points.some((p) => {
              getPointId(p) === getPointId(point);
            });
          });
          const updatedClusters = state.clusters.map((cluster) => ({
            ...cluster,
            points: cluster.points.filter((p) => getPointId(p) !== getPointId(point)),
          }));

          return {
            clusters: [
              ...updatedClusters,
              {
                ...newCluster,
                points: [
                  {
                    ...point,
                    original_cluster: oldCluster ?? null,
                    is_tagged: true,
                  },
                ],
              },
            ],
          };
        }),
    }),
    {
      name: 'dashboard-storage', // unique name for the storage
      storage: createJSONStorage(() => localStorage),
    },
  ),
);

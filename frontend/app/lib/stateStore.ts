import { create } from 'zustand';
import type { Point, Play } from '~/types/data';

// Define and export the PendingEdit interface
export interface PendingEdit {
  point_id: string; // Typically event_id
  game_id: string;
  new_cluster: string;
  original_cluster: string;
}

export interface DashboardState {
  selectedPlay: Point | null;
  pendingEdits: PendingEdit[];
  scatterPoints: Point[];
  updatePlay: (play: Point | null) => void;
  addPendingEdit: (edit: PendingEdit) => void;
  updatePendingEditTag: (pointId: string, newCluster: string) => void;
  clearPendingEdits: () => void;
  setScatterPoints: (points: Point[]) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  selectedPlay: null,
  pendingEdits: [],
  scatterPoints: [],
  updatePlay: (play) => set({ selectedPlay: play }),
  addPendingEdit: (edit) =>
    set((state) => ({
      // Avoid duplicates based on point_id and game_id
      pendingEdits: [
        ...state.pendingEdits.filter(
          (e) => !(e.point_id === edit.point_id && e.game_id === edit.game_id),
        ),
        edit,
      ],
    })),
  updatePendingEditTag: (pointId, newCluster) =>
    set((state) => ({
      pendingEdits: state.pendingEdits.map((edit) =>
        edit.point_id === pointId ? { ...edit, new_cluster: newCluster } : edit,
      ),
    })),
  clearPendingEdits: () => set({ pendingEdits: [] }),
  setScatterPoints: (points) => set({ scatterPoints: points }),
}));

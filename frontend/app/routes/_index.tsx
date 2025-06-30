import { useEffect } from 'react';
import { data, useLoaderData, useSearchParams, type ClientLoaderFunctionArgs } from 'react-router';
import ClusterView from '~/components/ClusterView';
import EmptyScatterGuide from '~/components/EmptyScatterGuide';
import { PlaysTable } from '~/components/PlaysTable';
import PlayView from '~/components/PlayView';
import ScatterPlot from '~/components/ScatterPlot';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import { Separator } from '~/components/ui/separator';
import { BASE_URL, GameFilter } from '~/lib/const';
import { fetchWrapper, purgeCacheOnGitCommitChange } from '~/lib/fetchCache';
import { useDashboardStore } from '~/lib/stateStore';
import { getPointId } from '~/lib/utils';
import type { ClusterData, Game, Team } from '~/types/data';
import type { Route } from './+types/_index';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  await purgeCacheOnGitCommitChange();
  const url = new URL(request.url);
  const teamID = url.searchParams.get('teamid');
  const timeframeUrl = url.searchParams.get('timeframe');
  const fetchScatter = url.searchParams.get('fetch_scatter');

  let timeframe =
    isNaN(Number(timeframeUrl)) || timeframeUrl === null
      ? GameFilter.LAST3
      : Number(url.searchParams.get('timeframe'));

  const bypassScatterCache = Boolean(fetchScatter);
  const state = useDashboardStore.getState();
  const promises = [];

  if (!state.teams.length) {
    promises.push(fetchWrapper<Team[]>(`${BASE_URL}/teams`).then((teams) => state.setTeams(teams)));
  }

  if (teamID && (bypassScatterCache || !state.clusters.length)) {
    promises.push(
      fetchWrapper<ClusterData[]>(
        `${BASE_URL}/teams/${teamID}/plays/scatter${timeframe ? `?timeframe=last_${timeframe}` : ''}`,
      ).then((data) => state.setClusters(data ?? [])),
    );
  }

  if (teamID && !state.games.length) {
    promises.push(
      fetchWrapper<Game[]>(`${BASE_URL}/teams/${teamID}/games`).then((games) =>
        state.setGames(games ?? []),
      ),
    );
  }
  await Promise.all(promises);

  return {
    timeframe,
    teamID,
  };
}

clientLoader.hydrate = true;

export default function Home() {
  const { teamID } = useLoaderData<typeof clientLoader>();
  const scatterData = useDashboardStore((state) => state.clusters);
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const clearSelectedPoint = useDashboardStore((state) => state.resetSelectedPoint);
  const clearSelectedCluster = useDashboardStore((state) => state.clearSelectedCluster);
  const [searchParams, setSearchParams] = useSearchParams();

  // Clear selectedPoint and selectedCluster when teamID is falsy or on mount
  useEffect(() => {
    if (!teamID) {
      clearSelectedPoint();
      clearSelectedCluster();
    }
  }, [teamID, clearSelectedPoint, clearSelectedCluster]);

  useEffect(() => {
    if (scatterData && searchParams.get('fetch_scatter')) {
      setSearchParams((prev) => {
        prev.delete('fetch_scatter');
        return prev;
      });
    }
  }, [searchParams, scatterData]);

  let tableData =
    selectedCluster && selectedPoint
      ? (
          scatterData?.find((d) => d.cluster_id === selectedCluster?.cluster_id)?.points ?? []
        ).filter((p) => getPointId(p) !== getPointId(selectedPoint))
      : [];
  const tableTitle = `Similar plays in cluster ${selectedCluster?.cluster_label ?? ''}`;
  return (
    <>
      <div className="space-y-4">
        <ResizablePanelGroup direction="horizontal" className="min-h-[500px]">
          <ResizablePanel id="left-panel" defaultSize={70}>
            {teamID ? <ScatterPlot /> : <EmptyScatterGuide />}
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={30}>
            <PlayView />
            <Separator orientation="horizontal" />
            <ClusterView />
          </ResizablePanel>
          <ResizableHandle />
        </ResizablePanelGroup>
      </div>
      <div className="mt-12 mb-16 space-y-6">
        {selectedCluster ? <PlaysTable data={tableData} title={tableTitle} /> : null}
      </div>
    </>
  );
}

import { useEffect } from 'react';
import {
  useLoaderData,
  useLocation,
  useSearchParams,
  type ClientLoaderFunctionArgs,
} from 'react-router';
import ClusterView from '~/components/ClusterView';
import EmptyScatterGuide from '~/components/EmptyScatterGuide';
import { PlaysTable } from '~/components/PlaysTable';
import PlayView from '~/components/PlayView';
import ScatterPlot from '~/components/ScatterPlot';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import { Separator } from '~/components/ui/separator';
import { BASE_URL, GameFilter } from '~/lib/const';
import {
  fetchWithCache,
  purgeCacheOnGitCommitChange,
  purgeScatterDataCache,
} from '~/lib/fetchCache';
import { useDashboardStore } from '~/lib/stateStore';
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

  const games = await (teamID
    ? fetchWithCache<Game[]>(`${BASE_URL}/teams/${teamID}/games`)
    : Promise.resolve(null));

  const totalGames = games?.length ?? 0;
  timeframe = Math.min(totalGames, timeframe);

  const bypassScatterCache = Boolean(fetchScatter);
  if (bypassScatterCache) purgeScatterDataCache(teamID);

  const fetchPromises: [Promise<Team[]>, Promise<ClusterData[] | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`),
    teamID
      ? fetchWithCache<ClusterData[]>(
          `${BASE_URL}/teams/${teamID}/plays/scatter${timeframe ? `?timeframe=last_${timeframe}` : ''}`,
          true,
          bypassScatterCache,
        )
      : Promise.resolve(null),
  ];

  const [teams, scatterData] = await Promise.all(fetchPromises);

  return {
    timeframe,
    teamID,
    totalGames,
    teams,
    games,
    scatterData,
  };
}

clientLoader.hydrate = true;

export default function Home() {
  const { scatterData: initialScatterData, teamID } = useLoaderData<typeof clientLoader>();
  const location = useLocation();
  const scatterData = useDashboardStore((state) => state.clusters);
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);

  useEffect(() => {
    if (initialScatterData) {
      const url = new URL(window.location.href);
      url.searchParams.delete('fetch_scatter');
      window.history.replaceState({}, '', url.toString());
    }
  }, [location, initialScatterData]);

  let tableData =
    scatterData?.find((d) => d.cluster_id === selectedCluster?.cluster_id)?.points ?? [];
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
      <div className="mt-12 mb-16 space-y-10">
        <PlaysTable data={tableData} title={tableTitle} />
      </div>
    </>
  );
}

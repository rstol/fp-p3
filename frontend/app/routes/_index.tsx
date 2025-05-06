import localforage from 'localforage';
import { useSearchParams, type ClientLoaderFunctionArgs } from 'react-router';
import ClusterView from '~/components/ClusterView';
import EmptyScatterGuide from '~/components/EmptyScatterGuide';
import { PlaysTable } from '~/components/PlaysTable';
import PlayView from '~/components/PlayView';
import ScatterPlot from '~/components/ScatterPlot';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import { Separator } from '~/components/ui/separator';
import { BASE_URL, GameFilter } from '~/lib/const';
import type { Point, Team } from '~/types/data';
import type { Route } from './+types/_index';
import { max } from 'd3';

interface ScatterDataResponse {
  total_games: number;
  points: Point[];
}

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

function createCacheKey(fullUrl: string, includeSearch: boolean = false): string {
  const urlObj = new URL(fullUrl);
  return includeSearch ? urlObj.pathname + urlObj.search : urlObj.pathname;
}

async function fetchWithCache<T>(url: string, includeSearch: boolean = false): Promise<T> {
  const key = createCacheKey(url, includeSearch);
  const cached = await localforage.getItem<T>(key);
  if (cached) {
    console.log(`Cache hit for ${key}`);
    return cached;
  }
  console.log(`Cache miss for ${key}`);

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch from ${url}`);

  const data: T = await res.json();
  await localforage.setItem(key, data);
  return data;
}
export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  const url = new URL(request.url);
  const teamID = url.searchParams.get('teamid');
  let timeframe = isNaN(Number(url.searchParams.get('timeframe')))
    ? GameFilter.LAST5
    : Number(url.searchParams.get('timeframe'));
  let totalGames = 0;

  if (teamID) {
    const response = await fetchWithCache<Team[]>(`${BASE_URL}/teams/${teamID}/games`);
    totalGames = response.length;
  }
  timeframe = Math.min(totalGames, timeframe);

  const fetchPromises: [Promise<Team[]>, Promise<ScatterDataResponse | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`),
    teamID
      ? fetchWithCache<ScatterDataResponse>(
          `${BASE_URL}/teams/${teamID}/plays/scatter${timeframe ? `?timeframe=last_${timeframe}` : ''}`,
          true,
        )
      : Promise.resolve(null),
  ];

  const [teams, scatterData] = await Promise.all(fetchPromises);
  return {
    timeframe,
    totalGames,
    teams,
    games: [], // unused
    scatterData,
  };
}

clientLoader.hydrate = true;

export default function Home() {
  const [searchParams] = useSearchParams();
  const teamID = searchParams.get('teamid');

  return (
    <>
      <div className="space-y-4">
        <ResizablePanelGroup direction="horizontal" className="min-h-[500px]">
          <ResizablePanel id="left-panel" defaultSize={70}>
            {teamID ? <ScatterPlot teamID={teamID} /> : <EmptyScatterGuide />}
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
        <PlaysTable title="More plays of the current cluster" />
        {/* <TeamsTable />
        <GamesTable /> */}
      </div>
    </>
  );
}

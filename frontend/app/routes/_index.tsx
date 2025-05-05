import localforage from 'localforage';
import { useSearchParams, type ClientLoaderFunctionArgs } from 'react-router';
import { useEffect } from 'react';
import { useLoaderData } from 'react-router-dom'; // Use react-router-dom for useLoaderData
import ClusterView from '~/components/ClusterView';
import EmptyScatterGuide from '~/components/EmptyScatterGuide';
import { PlaysTable } from '~/components/PlaysTable';
import PlayView from '~/components/PlayView';
import ScatterPlot from '~/components/ScatterPlot';
import { Button } from '~/components/ui/button';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import { Separator } from '~/components/ui/separator';
import { BASE_URL, GameFilter } from '~/lib/const';
import type { Point, Team } from '~/types/data';
import type { Route } from './+types/_index';
import { useDashboardStore } from '~/lib/stateStore';

interface ScatterDataResponse {
  total_games: number;
  points: Point[];
}

export function meta({ }: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  const url = new URL(request.url);
  console.log(url.pathname);
  const teamid = url.searchParams.get('teamid');
  const timeframe = url.searchParams.get('timeframe') ?? GameFilter.LAST3;
  const forceRefresh = url.searchParams.get('refresh') === 'true';

  function createCacheKey(fullUrl: string, includeSearch: boolean = false): string {
    const urlObj = new URL(fullUrl);
    return includeSearch ? urlObj.pathname + urlObj.search : urlObj.pathname;
  }

  async function fetchWithCache<T>(url: string, includeSearch: boolean = false, skipCache: boolean = false): Promise<T> {
    const key = createCacheKey(url, includeSearch);
    if (!skipCache) {
      const cached = await localforage.getItem<T>(key);
      if (cached) {
        console.log(`Cache hit for ${key}`);
        return cached;
      }
    }

    console.log(`Cache miss for ${key}`);

    const res = await fetch(url);
    if (!res.ok) {
      // Handle specific errors for debugging
      if (res.status === 404 && url.includes('/scatter')) {
        console.error(`Scatter data not found for team ${teamid} and timeframe ${timeframe}`);
        // Return null or an empty structure instead of throwing for 404 on scatter data
        // This allows the UI to show "No data" instead of crashing
        return { total_games: 0, points: [] } as unknown as T;
      }
      throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    await localforage.setItem(key, data);
    return data;
  }

  const fetchPromises: [Promise<Team[]>, Promise<ScatterDataResponse | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`, false, forceRefresh),
    teamid
      ? fetchWithCache<ScatterDataResponse>(
        `${BASE_URL}/teams/${teamid}/plays/scatter${timeframe ? `?timeframe=${timeframe}` : ''}`,
        true,
        forceRefresh
      )
      : Promise.resolve(null),
  ];

  const [teams, scatterDataResult] = await Promise.all(fetchPromises);

  // Log the fetched data for debugging
  console.log("Fetched Teams:", teams?.length);
  console.log("Fetched Scatter Data:", scatterDataResult);

  return {
    teams,
    games: [], // Still unused, but part of the original structure
    scatterData: scatterDataResult,
  };
}

clientLoader.hydrate = true;

async function clearDataCache() {
  await localforage.clear();
  // Keep existing search params but add refresh=true
  const currentSearchParams = new URLSearchParams(window.location.search);
  currentSearchParams.set('refresh', 'true');
  window.location.search = currentSearchParams.toString();
}

export default function Home() {
  const [searchParams] = useSearchParams();
  const teamID = searchParams.get('teamid');
  const { setScatterPoints } = useDashboardStore();

  const { scatterData } = useLoaderData<typeof clientLoader>();

  useEffect(() => {
    // console.log("Loader data received in component:", scatterData);
    if (scatterData?.points && scatterData.points.length > 0) {
      // console.log("Setting scatter points:", scatterData.points.length);
      setScatterPoints(scatterData.points);
    } else if (teamID) {
      // console.log("Clearing scatter points because teamID is set but no points received.");
      // If a team is selected but no points came back, clear the store
      setScatterPoints([]);
    }
    // No 'else' needed - if no teamID, we don't clear existing points
    // from a previous selection until a new team is selected and data loads (or fails to load).
  }, [scatterData, teamID, setScatterPoints]);

  return (
    <>
      <div className="space-y-4">
        <div className="flex justify-end px-4 py-2">
          <Button
            variant="outline"
            size="sm"
            onClick={clearDataCache}
            className="text-xs"
          >
            Refresh Data
          </Button>
        </div>
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

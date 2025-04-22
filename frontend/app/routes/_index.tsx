import localforage from 'localforage';
import { useSearchParams, type ClientLoaderFunctionArgs } from 'react-router';
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

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  const url = new URL(request.url);
  console.log(url.pathname);
  const teamid = url.searchParams.get('teamid');
  const timeframe = url.searchParams.get('timeframe') ?? GameFilter.LAST3;
  const forceRefresh = url.searchParams.get('refresh') === 'true';
  // invariant(typeof teamid === 'strin g', 'teamid is required');
  let scatterData: null | ScatterDataResponse = null;

  function createCacheKey(fullUrl: string, includeSearch: boolean = false): string {
    const urlObj = new URL(fullUrl);
    return includeSearch ? urlObj.pathname + urlObj.search : urlObj.pathname;
  }

  // Helper function for caching fetches
  async function fetchWithCache<T>(url: string, includeSearch: boolean = false, skipCache: boolean = false): Promise<T> {
    const key = createCacheKey(url, includeSearch);
    
    // Check if we should skip cache due to force refresh
    if (!skipCache) {
      const cached = await localforage.getItem<T>(key);
      if (cached) {
        console.log(`Cache hit for ${key}`);
        return cached;
      }
    }
    
    console.log(`Cache miss for ${key}`);

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch from ${url}`);

    const data: T = await res.json();
    await localforage.setItem(key, data);
    return data;
  }

  const fetchPromises: [Promise<Team[]>, Promise<ScatterDataResponse | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`),
    teamid
      ? fetchWithCache<ScatterDataResponse>(
          `${BASE_URL}/teams/${teamid}/plays/scatter${timeframe ? `?timeframe=${timeframe}` : ''}`,
          true,
          forceRefresh // Skip cache if forceRefresh is true
        )
      : Promise.resolve(null),
  ];

  const [teams, scatterDataResult] = await Promise.all(fetchPromises);
  scatterData = scatterDataResult;

  return {
    teams,
    games: [], // unused
    scatterData,
  };
}

clientLoader.hydrate = true;

// Function to clear cache and refresh data
async function clearDataCache() {
  await localforage.clear();
  window.location.href = window.location.pathname + '?refresh=true&' + 
    window.location.search.substring(1).replace(/&?refresh=true/g, '');
}

export default function Home() {
  const [searchParams] = useSearchParams();
  const teamID = searchParams.get('teamid');

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

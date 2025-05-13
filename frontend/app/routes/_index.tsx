import localforage from 'localforage';
import {
  useSearchParams,
  type ClientLoaderFunctionArgs,
  type ActionFunctionArgs,
} from 'react-router';
import ClusterView from '~/components/ClusterView';
import EmptyScatterGuide from '~/components/EmptyScatterGuide';
import { PlaysTable } from '~/components/PlaysTable';
import PlayView from '~/components/PlayView';
import ScatterPlot from '~/components/ScatterPlot';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import { Separator } from '~/components/ui/separator';
import { BASE_URL, GameFilter } from '~/lib/const';
import type { Game, Point, Team } from '~/types/data';
import type { Route } from './+types/_index';
import { fetchWithCache, purgeCacheIfNeeded } from '~/lib/fetchCache';

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

export async function clientAction({ request }: ActionFunctionArgs) {
  if (request.method !== 'POST') {
    return { success: false, error: 'Invalid request method - expected POST', status: 405 };
  }

  try {
    const payload = await request.json();
    const { teamId, updates } = payload;

    if (!teamId || typeof teamId !== 'string') {
      return { success: false, error: 'Missing or invalid teamId in payload', status: 400 };
    }
    if (!Array.isArray(updates) || updates.length === 0) {
      return { success: false, error: 'Missing or empty updates array in payload', status: 400 };
    }

    // Construct the full backend API URL using BASE_URL from constants
    const backendApiUrl = `${BASE_URL}/teams/${teamId}/plays/scatter`;

    const backendResponse = await fetch(backendApiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ updates }),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse
        .json()
        .catch(() => ({ message: 'Backend returned an error without JSON body' }));
      console.error(
        `[_index.clientAction] Backend API error: ${backendResponse.status}`,
        errorData,
      );
      return {
        success: false,
        error: errorData.message || 'Failed to apply changes via backend',
        status: backendResponse.status,
      };
    }

    const responseData = await backendResponse.json();
    return { success: true, data: responseData, message: 'Changes applied successfully.' };
  } catch (error) {
    console.error('[_index.clientAction] Error processing request:', error);
    if (error instanceof SyntaxError) {
      return { success: false, error: 'Invalid JSON payload provided', status: 400 };
    }
    return { success: false, error: 'An unexpected server error occurred', status: 500 };
  }
}

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  purgeCacheIfNeeded();
  const url = new URL(request.url);
  const teamID = url.searchParams.get('teamid');
  let timeframe = isNaN(Number(url.searchParams.get('timeframe')))
    ? GameFilter.LAST5
    : Number(url.searchParams.get('timeframe'));

  const games = await (teamID
    ? fetchWithCache<Game[]>(`${BASE_URL}/teams/${teamID}/games`)
    : Promise.resolve(null));
  const totalGames = games?.length ?? 0;
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
    games,
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

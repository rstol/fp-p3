import {
  useSearchParams,
  type ActionFunctionArgs,
  type ClientLoaderFunctionArgs,
} from 'react-router';
import ClusterView from '~/components/ClusterView';
import EmptyScatterGuide from '~/components/EmptyScatterGuide';
import { PlaysTable } from '~/components/PlaysTable';
import PlayView from '~/components/PlayView';
import ScatterPlot from '~/components/ScatterPlot';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import { Separator } from '~/components/ui/separator';
import { BASE_URL, GameFilter, PlayActions } from '~/lib/const';
import { fetchWithCache, purgeCacheOnGitCommitChange } from '~/lib/fetchCache';
import type { Cluster, Game, Team } from '~/types/data';
import type { Route } from './+types/_index';
import invariant from 'tiny-invariant';

type ScatterDataResponse = Cluster[];

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

  console.log(fetchScatter);

  const fetchPromises: [Promise<Team[]>, Promise<ScatterDataResponse | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`),
    teamID
      ? fetchWithCache<ScatterDataResponse>(
          `${BASE_URL}/teams/${teamID}/plays/scatter${timeframe ? `?timeframe=last_${timeframe}` : ''}`,
          true,
          Boolean(fetchScatter),
        )
      : Promise.resolve(null),
  ];

  const [teams, scatterData] = await Promise.all(fetchPromises);
  console.log(scatterData);
  return {
    timeframe,
    totalGames,
    teams,
    games,
    scatterData,
  };
}

export async function clientAction({ request }: ActionFunctionArgs) {
  invariant(request.method.toLowerCase() === 'post', 'Invalid request method - expected POST');

  const formData = await request.formData();

  const { _action, ...values } = Object.fromEntries(formData);
  invariant(_action && typeof _action === 'string', 'Missing or invalid action in payload');
  invariant(
    values.eventId && typeof values.eventId === 'string',
    'Missing or invalid eventId in payload',
  );
  invariant(
    values.gameId && typeof values.gameId === 'string',
    'Missing or invalid gameId in payload',
  );
  console.log(values);
  switch (_action) {
    case PlayActions.UpdateAllPlayFields: {
      let { note, clusters, gameId, eventId } = values;
      clusters = JSON.parse(clusters as string);
      invariant(
        Array.isArray(clusters) && clusters.length > 0,
        'Missing or invalid clusters in payload',
      );

      // Set cluster id to null for new cluster
      const cluster = clusters[0];
      const payload = {
        cluster_id: cluster.id.startsWith('new_cluster') ? null : cluster.id,
        cluster_name: cluster.text,
        note: note,
      };
      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      return backendResponse; // No actual data just 200 or 40x
    }
    case PlayActions.UpdatePlayNote: {
      const { note, gameId, eventId } = values;

      const payload = {
        note: note,
      };
      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      return backendResponse; // No actual data just 200 or 40x
    }
    case PlayActions.UpdatePlayCluster:
      let { clusters, gameId, eventId } = values;
      clusters = JSON.parse(clusters as string);
      invariant(
        Array.isArray(clusters) && clusters.length > 0,
        'Missing or invalid clusters in payload',
      );

      // Set cluster id to null for new cluster
      const cluster = clusters[0];
      const payload = {
        cluster_id: cluster.id.startsWith('new_cluster') ? null : cluster.id,
        cluster_name: cluster.text,
      };
      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      return backendResponse; // No actual data just 200 or 40x
    default:
      break;
  }
  return;
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
        <PlaysTable />
        {/* <TeamsTable />
        <GamesTable /> */}
      </div>
    </>
  );
}

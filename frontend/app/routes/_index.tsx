import {
  useLocation,
  useNavigation,
  useParams,
  useSearchParams,
  type ClientLoaderFunctionArgs,
} from 'react-router';
import { BASE_URL } from '~/lib/const';
import type { Game, Point, Team } from '~/types/data';
import type { Route } from './+types/_index';
import { Separator } from '~/components/ui/separator';
import ClusterView from '~/components/ClusterView';
import Filters from '~/components/Filters';
import { PlaysTable } from '~/components/PlaysTable';
import PlayView from '~/components/PlayView';
import ScatterPlot from '~/components/ScatterPlot';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '~/components/ui/resizable';
import EmptyScatterGuide from '~/components/EmptyScatterGuide';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  const url = new URL(request.url);
  const teamid = url.searchParams.get('teamid');
  // invariant(typeof teamid === 'string', 'teamid is required');
  let scatterData: null | Point[] = null;
  if (teamid) {
    const playScatterRes = await fetch(`${BASE_URL}/teams/${teamid}/plays/scatter`);
    if (!playScatterRes.ok) throw new Error('Failed to fetch scatter data');
    scatterData = await playScatterRes.json();
  }

  const [teamRes, gameRes] = await Promise.all([
    fetch(`${BASE_URL}/teams`),
    fetch(`${BASE_URL}/games`),
  ]);

  if (!teamRes.ok) throw new Error('Failed to fetch team');
  if (!gameRes.ok) throw new Error('Failed to fetch games');

  const [teams, games]: [Team[], Game[]] = await Promise.all([teamRes.json(), gameRes.json()]);
  return {
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
        <Filters teamID={teamID} />
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

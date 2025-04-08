import { type ClientLoaderFunctionArgs } from 'react-router';
import ScatterPlot from '~/components/ScatterPlot';
import TeamsTable from '~/components/TeamsTable';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import { BASE_URL } from '~/lib/const';
import type { Team, Game } from '~/types/data';
import type { Route } from './+types/home';
import GamesTable from '~/components/GamesTable';
import Header from '~/components/Header';
import Filters from '~/components/Filters';
import PlayView from '~/components/PlayView';
import ClusterView from '~/components/ClusterView';
import { PlaysTable } from '~/components/PlaysTable';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
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
  };
}

clientLoader.hydrate = true;

function MainView() {
  return (
    <>
      <div className="space-y-4">
        <Filters />
        <ResizablePanelGroup direction="horizontal" className="min-h-[500px]">
          <ResizablePanel id="left-panel" defaultSize={70}>
            <ScatterPlot />
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={30}>
            <PlayView />
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

function TaggedPlaysView() {
  return (
    <div className="mt-6">
      <PlaysTable title="Tagged plays" />
    </div>
  );
}

export default function Home() {
  return (
    <div className="mx-auto max-w-[1440px] px-4 py-6">
      <div className="space-y-2">
        <Header
          tabs={[
            { title: 'Play Analyzer', children: <MainView /> },
            { title: 'Tagged Plays', children: <TaggedPlaysView /> },
          ]}
        />
      </div>
    </div>
  );
}

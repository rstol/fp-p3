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
    <div className="space-y-4">
      <Filters />
      <ResizablePanelGroup direction="horizontal" className="min-h-[500px]">
        <ResizablePanel id="left-panel" defaultSize={70}>
          <ScatterPlot />
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={30}>
          <div className="flex h-full items-center justify-center p-6">
            <span className="font-semibold">Basketball Court Play view</span>
            {/* TODO render play  */}
          </div>
        </ResizablePanel>
        <ResizableHandle />
      </ResizablePanelGroup>
    </div>
  );
}

function TaggedPlayView() {
  return <div>Tagged Plays</div>;
}

export default function Home() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <div className="space-y-2">
        <Header
          tabs={[
            { title: 'Play Exploration', children: <MainView /> },
            { title: 'Tagged Plays', children: <TaggedPlayView /> },
          ]}
        />
      </div>
      <div className="mt-10 space-y-10">
        <TeamsTable />
        <GamesTable />
      </div>
    </div>
  );
}

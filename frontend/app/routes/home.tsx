import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '~/components/ui/resizable';
import type { Route } from './+types/home';
import ScatterPlot from '~/components/ScatterPlot';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

export default function Home() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <h1>Dashboard</h1>
      <ResizablePanelGroup direction="horizontal" className="min-h-[500px]">
        <ResizablePanel defaultSize={70}>
          <ScatterPlot />
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={30}>
          <div className="flex h-full items-center justify-center p-6">
            <span className="font-semibold">Basketball Play view</span>
          </div>
        </ResizablePanel>
        <ResizableHandle />
      </ResizablePanelGroup>
    </div>
  );
}

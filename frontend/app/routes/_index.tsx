import { useEffect, useRef } from 'react';
import { useLoaderData, useSearchParams, type ClientLoaderFunctionArgs } from 'react-router';
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
import introJs from 'intro.js';
import { useDashboardStore } from '~/lib/stateStore';
import { getPointId } from '~/lib/utils';
import type { ClusterData, Game, Team } from '~/types/data';
import type { Route } from './+types/_index';
import { setCookie, getCookie } from '~/lib/cookies';
import type { IntroJs } from 'intro.js/src/intro';

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
  if (bypassScatterCache) {
    await purgeScatterDataCache(teamID);
  }

  const fetchPromises: [Promise<Team[]>, Promise<ClusterData[] | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`),
    teamID
      ? fetchWithCache<ClusterData[]>(
          `${BASE_URL}/teams/${teamID}/plays/scatter${timeframe ? `?timeframe=last_${timeframe}` : ''}`,
          true,
          bypassScatterCache,
          true,
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
  const scatterData = useDashboardStore((state) => state.clusters);
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const clearSelectedPoint = useDashboardStore((state) => state.resetSelectedPoint);
  const clearSelectedCluster = useDashboardStore((state) => state.clearSelectedCluster);
  const [searchParams, setSearchParams] = useSearchParams();

  // Store tour instance and started flag
  const tourRef = useRef<any>(null);
  const tourStarted = useRef(false);

  useEffect(() => {
    // Check if tour has been shown
    if (getCookie('introjs-tour-shown') || tourStarted.current) {
      // console.log('Tour skipped: cookie or tourStarted');
      return;
    }

    // Wait for table to render
    const startTour = () => {
      // Prevent starting if tour already exists
      if (tourRef.current) {
        // console.log('Tour already exists, skipping start');
        return;
      }

      const tour = introJs();
      tourRef.current = tour;
      tourStarted.current = true;

      tour.setOptions({
        steps: [
          {
            title: 'Welcome to DeepPlaybook, the deep-learning based Basketball Play-by-Play Analysis tool!',
            intro: 'Let’s get started with a quick tour.',
            position: 'bottom',
          },
          {
            element: '#left-panel',
            intro: 'Please select a team to analyze.',
            position: 'right',
          },
          {
            element: '#left-panel',
            intro: 'Please select a play to analyze by clicking on one of the points.',
            position: 'right',
          },
          {
            element: '#r${selectedPoint.event_id}',
            intro: 'This is the play overview.',
            position: 'right',
          },
          {
            element: '#scatter-plot', // Example: next step after team selection
            intro: 'This is the scatter plot showing play analysis.',
            position: 'left',
          },
        ],
        showProgress: true,
        doneLabel: 'Finish',
        dontShowAgain: true,
        dontShowAgainLabel: 'Don’t show again',
        dontShowAgainCookie: 'introjs-tour-shown',
        dontShowAgainCookieDays: 365,
      });

      // Block progression at team selection step until teamid is in searchParams
      tour.onbeforechange(function () {
        const currentStep = this._currentStep;
        if (currentStep === 2) { // Step 2 (team selection, zero-based index)
          return !!searchParams.get('teamid'); // Proceed only if teamid exists
        }
        if (currentStep === 3) {
          return !!selectedPoint; // Proceed only if selectedPoint exists
        }
        return true; // Allow other steps to proceed
      });

      tour.oncomplete(() => {
        setCookie('introjs-tour-shown', 'true', 365);
        tourRef.current = null;
        tourStarted.current = false;
      });

      tour.onexit(() => {
        setCookie('introjs-tour-shown', 'true', 365);
        tourRef.current = null;
        tourStarted.current = false;
      });

      tour.start();
    };

    // Wait for #teams-table to be rendered
    setTimeout(startTour, 500);
    
// Cleanup on unmount
    return () => {
      if (tourRef.current) {
        console.log('Cleaning up tour on unmount');
        tourRef.current.exit();
        tourRef.current = null;
        tourStarted.current = false;
      }
    };
  }, [searchParams]);

  // Separate effect to handle team and play selection
  useEffect(() => {
    if (tourRef.current) {
      const currentStep = tourRef.current._currentStep;
      console.log('Selection check, step:', currentStep, 'teamid:', searchParams.get('teamid'), 'selectedPoint:', !!selectedPoint);
      if (currentStep === 2 && searchParams.get('teamid')) {
        console.log('Advancing to play selection step');
        tourRef.current.nextStep();
      } else if (currentStep === 3 && selectedPoint) {
        console.log('Advancing to play overview step');
        tourRef.current.nextStep();
      }
    }
  }, [searchParams, selectedPoint]);

  // Clear selectedPoint and selectedCluster when teamID is falsy or on mount
  useEffect(() => {
    if (!teamID) {
      clearSelectedPoint();
      clearSelectedCluster();
    }
  }, [teamID, clearSelectedPoint, clearSelectedCluster]);

  useEffect(() => {
    if (initialScatterData && searchParams.get('fetch_scatter')) {
      setSearchParams((prev) => {
        prev.delete('fetch_scatter');
        return prev;
      });
    }
  }, [searchParams, initialScatterData]);

  let tableData =
    selectedCluster && selectedPoint
      ? (
          scatterData?.find((d) => d.cluster_id === selectedCluster?.cluster_id)?.points ?? []
        ).filter((p) => getPointId(p) !== getPointId(selectedPoint))
      : [];
  const tableTitle = `Similar plays in cluster ${selectedCluster?.cluster_label ?? ''}`;
  return (
    <>
      <div className="space-y-4" id="main-index">
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
      <div className="mt-12 mb-16 space-y-6">
        {selectedCluster ? <PlaysTable data={tableData} title={tableTitle} /> : null}
      </div>
    </>
  );
}

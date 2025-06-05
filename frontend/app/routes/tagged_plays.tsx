import { useLoaderData, type ClientLoaderFunctionArgs } from 'react-router';
import { PlaysTable } from '~/components/PlaysTable';
import { BASE_URL, GameFilter } from '~/lib/const';
import { fetchWithCache } from '~/lib/fetchCache';
import { useDashboardStore } from '~/lib/stateStore';
import type { ClusterData, Game, Team } from '~/types/data';

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  const url = new URL(request.url);
  const teamID = url.searchParams.get('teamid');
  const timeframeUrl = url.searchParams.get('timeframe');
  let timeframe =
    isNaN(Number(timeframeUrl)) || timeframeUrl === null
      ? GameFilter.LAST3
      : Number(url.searchParams.get('timeframe'));

  if (!teamID) return {};

  const fetchPromises: [Promise<Team[]>, Promise<Game[]>, Promise<ClusterData[] | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`),
    fetchWithCache<Game[]>(`${BASE_URL}/teams/${teamID}/games`),
    fetchWithCache<ClusterData[]>(
      `${BASE_URL}/teams/${teamID}/plays/scatter${timeframe ? `?timeframe=last_${timeframe}` : ''}`,
      true,
    ),
  ];
  const [teams, games, scatterData] = await Promise.all(fetchPromises);
  return {
    teamID,
    teams,
    games,
    scatterData,
  };
}

clientLoader.hydrate = true;

export default function TaggedPlaysView() {
  const scatterData = useDashboardStore((state) => state.clusters);
  const loaderData = useLoaderData<typeof clientLoader>();
  const clusters = scatterData?.length ? scatterData : loaderData.scatterData;
  const hasTaggedPlays = clusters?.some((cluster) => cluster.points.some((p) => p.is_tagged));
  return (
    <div className="mt-6">
      {hasTaggedPlays
        ? clusters?.map((cluster) => {
            const points = cluster.points.filter((p) => p.is_tagged);
            return points.length ? (
              <PlaysTable
                key={cluster.cluster_id}
                data={points}
                title={`Tagged plays in cluster ${cluster.cluster_label ?? ''}`}
              />
            ) : null;
          })
        : 'No plays tagged'}
    </div>
  );
}

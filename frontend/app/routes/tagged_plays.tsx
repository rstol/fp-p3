import type { ClientLoaderFunctionArgs } from 'react-router';
import { PlaysTable } from '~/components/PlaysTable';
import { BASE_URL } from '~/lib/const';
import { fetchWithCache } from '~/lib/fetchCache';
import { useDashboardStore } from '~/lib/stateStore';
import type { Game, Team } from '~/types/data';

export async function clientLoader({ request }: ClientLoaderFunctionArgs) {
  const url = new URL(request.url);
  const teamID = url.searchParams.get('teamid');

  const fetchPromises: [Promise<Team[]>, Promise<Game[] | null>] = [
    fetchWithCache<Team[]>(`${BASE_URL}/teams`),
    fetchWithCache<Game[]>(`${BASE_URL}/teams/${teamID}/games`),
  ];

  const [teams, games] = await Promise.all(fetchPromises);

  return {
    teamID,
    teams,
    games,
  };
}

clientLoader.hydrate = true;

export default function TaggedPlaysView() {
  const scatterData = useDashboardStore((state) => state.clusters);
  console.log(scatterData);
  return (
    <div className="mt-6">
      {scatterData.length
        ? (scatterData.map((cluster) => {
            const title = `Tagged plays in cluster ${cluster?.cluster_label ?? ''}`;
            const points = cluster.points.filter((p) => p.is_tagged);
            if (!points.length) return null;
            return <PlaysTable key={cluster.cluster_id} data={points} title={title} />;
          }) ?? 'No points tagged')
        : 'No plays tagged'}
    </div>
  );
}

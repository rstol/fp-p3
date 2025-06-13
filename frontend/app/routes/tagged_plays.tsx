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
  const tags = useDashboardStore((state) => state.tags);
  const hasTaggedPlays = tags?.some((tag) => tag.points.some((p) => p.is_tagged));

  return (
    <div className="mt-6">
      {!hasTaggedPlays
        ? tags?.map((tag) => {
            return tag?.points.length ? (
              <PlaysTable
                key={tag.tag_label}
                data={tag.points}
                title={`Plays tagged with ${tag.tag_label}`}
              />
            ) : null;
          })
        : 'No plays tagged'}
    </div>
  );
}

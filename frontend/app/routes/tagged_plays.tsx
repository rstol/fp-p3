import { useLoaderData, type ClientLoaderFunctionArgs } from 'react-router';
import { PlaysTable } from '~/components/PlaysTable';
import { BASE_URL, GameFilter } from '~/lib/const';
import { fetchWithCache } from '~/lib/fetchCache';
import { useDashboardStore } from '~/lib/stateStore';
import type { ClusterData, Game, Point, Tag, Team } from '~/types/data';

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
  const clusters = useDashboardStore((state) => state.clusters);
  const hasTaggedPlays = clusters?.some((c) => c.points.some((p) => p?.tags?.length));

  const groupedByTag = clusters
    .flatMap((cluster) => cluster.points)
    .reduce<Record<string, { tag: Tag; points: Point[] }>>((acc, point) => {
      (point.tags || []).forEach((tag) => {
        const key = tag.tag_id;
        if (!acc[key]) {
          acc[key] = { tag, points: [] };
        }
        acc[key].points.push(point);
      });
      return acc;
    }, {});
  const tagGroups = Object.values(groupedByTag);

  return (
    <div className="mt-6">
      {hasTaggedPlays
        ? tagGroups?.map((tagGroup) => {
            return tagGroup.points.length ? (
              <PlaysTable
                key={tagGroup.tag.tag_label}
                data={tagGroup.points}
                title={`Plays tagged with ${tagGroup.tag.tag_label}`}
              />
            ) : (
              'No plays tagged'
            );
          })
        : 'No plays tagged'}
    </div>
  );
}

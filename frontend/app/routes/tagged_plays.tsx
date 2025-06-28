import { PlaysTable } from '~/components/PlaysTable';
import { useDashboardStore } from '~/lib/stateStore';
import type { Point, Tag } from '~/types/data';

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

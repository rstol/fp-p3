import { PlaysTable } from '~/components/PlaysTable';
import { useDashboardStore } from '~/lib/stateStore';

export default function TaggedPlaysView() {
  const tags = useDashboardStore((state) => state.tags);
  const hasTaggedPlays = tags?.some((tag) => tag.points.some((p) => p.is_tagged));

  return (
    <div className="mt-6">
      {hasTaggedPlays
        ? tags?.map((tag) => {
            const points = tag.points;
            return points.length ? (
              <PlaysTable
                key={tag.tag_label}
                data={points}
                title={`Plays tagged with ${tag.tag_label}`}
              />
            ) : null;
          })
        : 'No plays tagged'}
    </div>
  );
}

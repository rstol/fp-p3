import { PlaysTable } from '~/components/PlaysTable';

export default function TaggedPlaysView() {
  // TODO group by cluster and only show tagged plays
  return (
    <div className="mt-6">
      <PlaysTable />
    </div>
  );
}

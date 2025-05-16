import { useDashboardStore } from '~/lib/stateStore';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ClusterDetailsSkeleton } from './LoaderSkeletons';

export default function ClusterView() {
  // TODO get this info for the currently selected cluster
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const isLoading = false; // TODO fetching data
  if (isLoading) return <ClusterDetailsSkeleton />;
  if (!selectedPoint) {
    return (
      <Card className="gap-4 border-none pt-1 shadow-none">
        <CardHeader>
          <CardTitle>Cluster Statistics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-sm">No cluster selected.</div>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card className="gap-4 border-none pb-1 shadow-none">
      <CardHeader>
        <CardTitle>Cluster {selectedPoint?.cluster} Statistics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="divide-y divide-solid">
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Points per Position:</span>
            <span className="flex-1 text-right">2.3</span>
          </div>
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Usage:</span>
            <span className="flex-1 text-right">17.68%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

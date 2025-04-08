import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './ui/card';

export default function ClusterView() {
  // TODO get this info for the currently selected cluster
  return (
    <Card className="gap-4 border-none shadow-none">
      <CardHeader>
        <CardTitle>Cluster XY Statistics</CardTitle>
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

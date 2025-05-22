import { zodResolver } from '@hookform/resolvers/zod';
import { Check } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { BASE_URL } from '~/lib/const';
import { useDashboardStore } from '~/lib/stateStore';
import { ClusterDetailsSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from './ui/form';
import { Input } from './ui/input';

const FormSchema = z.object({
  clusterLabel: z.string(),
  clusterId: z.string(),
});

export default function ClusterView() {
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);
  const updateSelectedCluster = useDashboardStore((state) => state.updateSelectedCluster);
  console.log(selectedCluster);
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      clusterLabel: selectedCluster?.cluster_label ?? '',
      clusterId: selectedCluster?.cluster_id,
    },
  });

  const isLoading = false; // TODO fetching data
  if (isLoading) return <ClusterDetailsSkeleton />;
  if (!selectedCluster) {
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

  async function onSubmit(data: z.infer<typeof FormSchema>) {
    console.log('submit', data);

    const payload = {
      cluster_label: data.clusterLabel,
    };
    await fetch(`${BASE_URL}/cluster/${data.clusterId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    updateSelectedCluster({ cluster_id: data.clusterId, cluster_label: data.clusterLabel });
  }

  return (
    <Card className="gap-4 border-none pb-1 shadow-none">
      <CardHeader>
        <CardTitle>Cluster {selectedCluster.cluster_label}</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <Form {...form}>
          <form className="grid w-full items-center gap-2" onSubmit={form.handleSubmit(onSubmit)}>
            <FormField
              control={form.control}
              name="clusterLabel"
              render={({ field }) => (
                <FormItem className="flex flex-col items-start">
                  <FormLabel className="text-sm">Change Cluster Label</FormLabel>
                  <div className="flex w-full gap-2">
                    <FormControl>
                      <Input type="text" placeholder="Add cluster label" {...field} />
                    </FormControl>
                    <Button type="submit" size="sm" className="h-9 w-10">
                      <Check size={6} />
                    </Button>
                  </div>
                  <FormMessage />
                </FormItem>
              )}
            />
          </form>
        </Form>
        <div>
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
        </div>
      </CardContent>
    </Card>
  );
}

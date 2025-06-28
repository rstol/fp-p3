import { zodResolver } from '@hookform/resolvers/zod';
import { Check, Loader2 } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { BASE_URL } from '~/lib/const';
import { useDashboardStore } from '~/lib/stateStore';
import { ClusterDetailsSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from './ui/form';
import { Input } from './ui/input';
import { useLoaderData } from 'react-router';
import type { clientLoader } from '~/routes/_index';
import type { ClusterMetadata } from '~/types/data';
import { useEffect, useState } from 'react';

function createFormSchema(existingLabels: string[]) {
  return z.object({
    clusterLabel: z
      .string()
      .min(1, 'Label cannot be empty')
      .refine((label) => !existingLabels.includes(label), {
        message: 'Label must be unique',
      }),
    clusterId: z.string(),
  });
}

function ClusterLabelForm({
  clusters,
  selectedCluster,
}: {
  clusters: ClusterMetadata[];
  selectedCluster: ClusterMetadata;
}) {
  const updateSelectedCluster = useDashboardStore((state) => state.updateSelectedCluster);
  const updateClusterLabel = useDashboardStore((state) => state.updateClusterLabel);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { teamID } = useLoaderData<typeof clientLoader>();
  const existingLabels = clusters
    .map((c) => c.cluster_label)
    .filter((label) => label && label !== selectedCluster?.cluster_label); // allow current label

  const FormSchema = createFormSchema(existingLabels as string[]);
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      clusterLabel: selectedCluster?.cluster_label ?? '',
      clusterId: selectedCluster?.cluster_id,
    },
  });

  useEffect(() => {
    form.reset({
      clusterLabel: selectedCluster?.cluster_label ?? '',
      clusterId: selectedCluster?.cluster_id,
    });
  }, [selectedCluster, form]);

  async function onSubmit(data: z.infer<typeof FormSchema>) {
    setIsSubmitting(true);

    const payload = {
      cluster_label: data.clusterLabel,
    };
    await fetch(`${BASE_URL}/teams/${teamID}/cluster/${data.clusterId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    updateClusterLabel(data.clusterId, data.clusterLabel);
    updateSelectedCluster({ cluster_id: data.clusterId, cluster_label: data.clusterLabel });
    setIsSubmitting(false);
  }
  return (
    <Form {...form}>
      <form className="grid w-full items-center gap-2" onSubmit={form.handleSubmit(onSubmit)}>
        <FormField
          control={form.control}
          name="clusterLabel"
          render={({ field }) => (
            <FormItem className="flex flex-col items-start">
              <FormLabel className="text-sm">Rename Cluster</FormLabel>
              <div className="flex w-full gap-2">
                <FormControl>
                  <Input type="text" placeholder="Add cluster label" {...field} />
                </FormControl>
                <Button disabled={isSubmitting} type="submit" size="sm" className="h-9 w-10">
                  {isSubmitting ? <Loader2 className="animate-spin" /> : <Check size={6} />}
                </Button>
              </div>
              <FormMessage />
            </FormItem>
          )}
        />
      </form>
    </Form>
  );
}

export default function ClusterView() {
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);
  const clustersState = useDashboardStore((state) => state.clusters);
  const data = useLoaderData<typeof clientLoader>();
  const clusterData = clustersState ?? data?.scatterData ?? [];
  const clusters = clusterData.map(({ cluster_id, cluster_label }) => ({
    cluster_id,
    cluster_label,
  }));

  const isLoading = false; // TODO fetching data
  if (isLoading) return <ClusterDetailsSkeleton />;
  if (!selectedCluster) {
    return (
      <Card className="gap-4 border-none pt-1 shadow-none">
        <CardHeader>
          <CardTitle>Cluster Statistics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-sm">No point selected.</div>
        </CardContent>
      </Card>
    );
  }

  const allPoints = data?.scatterData?.flatMap((c) => c.points);
  const currentCluster = clusterData.find((c) => c.cluster_id === selectedCluster.cluster_id);
  const usage =
    currentCluster && allPoints ? (100 / allPoints.length) * currentCluster.points.length : 0;
  const misses = currentCluster?.points.filter((p) => p.event_type === 2);
  const makes = currentCluster?.points.filter((p) => p.event_type === 1);
  const makeMissRatio =
    misses && makes ? (100 * makes?.length) / (makes.length + misses?.length) : 0;

  return (
    <Card className="gap-4 border-none pb-1 shadow-none">
      <CardHeader>
        <CardTitle>Cluster {selectedCluster.cluster_label}</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <ClusterLabelForm clusters={clusters} selectedCluster={selectedCluster} />
        <div>
          <div className="divide-y divide-solid">
            <div className="flex gap-4 pb-1">
              <span className="shrink-0">Usage</span>
              <span className="flex-1 text-right">{usage.toFixed(1)}%</span>
            </div>
            <div className="py-1">
              <div className="flex gap-4">
                <span className="shrink-0">Success</span>
                <span className="flex-1 text-right">{makeMissRatio.toFixed(1)}%</span>
              </div>
              <small>made shot vs. missed shot</small>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

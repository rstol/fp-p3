import { useDashboardStore } from '~/lib/stateStore';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ClusterDetailsSkeleton } from './LoaderSkeletons';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { Check } from 'lucide-react';
import { Button } from './ui/button';
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from './ui/form';
import { Input } from './ui/input';
import { useSubmit } from 'react-router';

const FormSchema = z.object({
  clusterLabel: z.string(),
  clusterId: z.string(),
});

export default function ClusterView() {
  // TODO get this info for the currently selected cluster
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      clusterLabel: selectedPoint?.cluster,
      clusterId: selectedPoint?.cluster, // TODO change this
    },
  });
  let submit = useSubmit();

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

  function onSubmit(data: z.infer<typeof FormSchema>) {
    console.log('submit', data);
    submit(data, {
      action: '/resources/cluster',
      method: 'post',
    });
  }

  return (
    <Card className="gap-4 border-none pb-1 shadow-none">
      <CardHeader>
        <CardTitle>Cluster {selectedPoint?.cluster}</CardTitle>
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

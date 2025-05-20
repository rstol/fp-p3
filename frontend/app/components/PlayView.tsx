import { zodResolver } from '@hookform/resolvers/zod';
import { type Tag, TagInput } from 'emblor';
import { Check } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useFetcher, useLoaderData, useSearchParams, useSubmit } from 'react-router';
import { z } from 'zod';
import { BASE_URL, EventType, PlayActions } from '~/lib/const';
import { useDashboardStore } from '~/lib/stateStore';
import type { clientLoader } from '~/routes/_index';
import type { Team } from '~/types/data';
import { clientAction } from '../routes/play';
import { PlayDetailsSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from './ui/form';
import { Input } from './ui/input';
// interface ClientActionResult {
//   success: boolean;
//   error?: string;
//   message?: string;
//   data?: any; // Consider a more specific type for 'data' if known
// }
const FormSchema = z.object({
  clusters: z.array(
    z.object({
      id: z.string(),
      text: z.string(),
    }),
  ),
  note: z.string().optional(),
});

export type PlayPayload = {
  data: z.infer<typeof FormSchema> & { eventId?: string; gameId?: string };
  action: PlayActions;
};

function PlayForm({ playDetails }: { playDetails: PlayDetails | null }) {
  const data = useLoaderData<typeof clientLoader>();
  const clusterData = data?.scatterData?.points ?? [];
  const clusters = Array.from(new Set(clusterData.map((d) => String(d.cluster)))).sort();
  const stageSelectedPlayClusterUpdate = useDashboardStore(
    (state) => state.stageSelectedPlayClusterUpdate,
  );

  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  // TODO temporary transform: use data schema
  const initialCluster = {
    id: String(selectedPoint?.cluster),
    text: String(selectedPoint?.cluster),
  };
  const initialTags = [...clusters].map((c) => ({ id: c, text: c }));
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      clusters: [initialCluster],
      note: '', //playDetails.note TODO
    },
  });
  const [tags, setTags] = useState<Tag[]>([initialCluster]);
  const [activeTagIndex, setActiveTagIndex] = useState<number | null>(null);
  const { setValue } = form;
  let submit = useSubmit();

  function onSubmit(data: z.infer<typeof FormSchema>) {
    console.log('submit', data);
    const payload = {
      action: PlayActions.UpdatePlayFields,
      data: JSON.stringify({ eventId: selectedPoint?.event_id, gameId: selectedPoint?.game_id, ...data }),
    };

    submit(payload, {
      action: '/play',
      method: 'post'
    });

    stageSelectedPlayClusterUpdate(data.clusters[0].id); // TODO
  }

  //TODO define for to new cluster data format
  const generateTagId = () => {
    const generatedId = Math.random() * 16;
    return `new_cluster_${generatedId}`;
  };

  return (
    <>
      <Form {...form}>
        <form className="grid w-full items-center gap-2" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="clusters"
            render={({ field }) => (
              <FormItem className="flex flex-col items-start">
                <FormLabel className="text-sm">Play Cluster</FormLabel>
                <div className="flex w-full gap-2">
                  <FormControl>
                    <TagInput
                      {...field}
                      autocompleteOptions={initialTags}
                      maxTags={1}
                      tags={tags}
                      inlineTags
                      addTagsOnBlur
                      styleClasses={{
                        input: 'focus-visible:outline-none shadow-none w-full',
                        tag: { body: 'h-7' },
                      }}
                      generateTagId={generateTagId}
                      enableAutocomplete
                      placeholder="Select or create cluster"
                      setTags={(newTags) => {
                        setTags(newTags);
                        setValue('clusters', newTags as [Tag, ...Tag[]]);
                      }}
                      activeTagIndex={activeTagIndex}
                      setActiveTagIndex={setActiveTagIndex}
                    />
                  </FormControl>
                  <Button onClick={form.handleSubmit(onSubmit)} size="sm" className="h-9 w-10">
                    <Check size={6} />
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="note"
            render={({ field }) => (
              <FormItem className="flex flex-col items-start">
                <FormLabel className="text-sm">Play Note</FormLabel>
                <div className="flex w-full gap-2">
                  <FormControl>
                    <Input type="text" placeholder="Add a note..." {...field} />
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
    </>
  );
}

export type PlayDetails = {
  game_id: string;
  event_id: string;
  event_type?: number;
  event_score?: string;
  possession_team_id?: number;
  event_desc_home?: string;
  event_desc_away?: string;
  period?: number;
  videoURL?: string;
  offenseTeam?: Team;
};

export default function PlayView() {
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const stagedChangesCount = useDashboardStore((state) => state.stagedChangesCount);
  const pendingClusterUpdates = useDashboardStore((state) => state.pendingClusterUpdates);
  const selectedTeamId = useDashboardStore((state) => state.selectedTeamId);
  const clearPendingClusterUpdates = useDashboardStore((state) => state.clearPendingClusterUpdates);
  const [playDetails, setPlayDetails] = useState<PlayDetails | null>(null);
  const [isLoadingPlayDetails, seIsLoadingPlayDetails] = useState(false);
  const [_, setSearchParams] = useSearchParams();
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  let submit = useSubmit();
  const clusterDataFetcher = useFetcher<typeof clientAction>();
  // TODO loader during updating the clusters
  const isUpdating = clusterDataFetcher.state !== 'idle';

  useEffect(() => {
    if (!selectedPoint) return;

    seIsLoadingPlayDetails(true);
    const fetchPlayDetails = async () => {
      try {
        let publicVideoPath = `videos/00${selectedPoint.game_id}/${selectedPoint.event_id}.mp4`;

        // Try to fetch from public folder first
        const checkPublicVideo = await fetch(publicVideoPath, { method: 'HEAD' }).catch(() => ({
          ok: false,
        }));

        const resDetails = await fetch(
          `${BASE_URL}/plays/00${selectedPoint.game_id}/${selectedPoint.event_id}`,
        );
        if (!resDetails.ok) throw new Error();
        const playDetails = await resDetails.json();

        if (!checkPublicVideo.ok) {
          const res = await fetch(
            `${BASE_URL}/plays/00${selectedPoint.game_id}/${selectedPoint.event_id}/video`,
          );
          const arrayBuffer = await res.arrayBuffer();
          const blob = new Blob([arrayBuffer], { type: 'video/mp4' });
          publicVideoPath = URL.createObjectURL(blob);
        }

        const offenseTeam = teams.find(
          (team) => String(team.teamid) === String(playDetails.possession_team_id),
        );
        setPlayDetails({ ...playDetails, videoURL: publicVideoPath, offenseTeam });
      } catch (error) {
        console.error('Failed to fetch play details:', error);
        // TODO update UI
        throw Error(`Failed to fetch play details: ${error}`);
      } finally {
        seIsLoadingPlayDetails(false);
      }
    };

    fetchPlayDetails();
  }, [selectedPoint]);

  if (!selectedPoint) {
    return (
      <Card className="gap-4 border-none pt-1 shadow-none">
        <CardHeader>
          <CardTitle>Selected Play Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm">No play selected.</div>
        </CardContent>
      </Card>
    );
  }

  const onClickApplyChanges = async () => {
    // Fetch with query param fetch_scatter set to bypass cache
    setSearchParams((prev) => {
      prev.set('fetch_scatter', 'True');
      return prev;
    });
    // TODO fix cluster update hook
    clearPendingClusterUpdates();
  };

  if (isLoadingPlayDetails) {
    return <PlayDetailsSkeleton />;
  }

  return (
    <Card className="gap-4 border-none pt-1 shadow-none">
      <CardHeader>
        <CardTitle>Selected Play Details</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {playDetails?.videoURL && (
          <video
            key={playDetails.videoURL} // Force remount component on change
            controls
            autoPlay
            disablePictureInPicture
            disableRemotePlayback
            loop
            muted
          >
            <source src={playDetails.videoURL} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        )}
        <div className="divide-y divide-solid text-sm">
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Offense Team</span>
            <span className="flex-1 text-right">{playDetails?.offenseTeam?.name}</span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Outcome</span>
            <span className="flex-1 text-right">
              {playDetails?.event_type ? EventType[playDetails.event_type] : 'N/A'}
            </span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Game Date</span>
            <span className="flex-1 text-right">
              {playDetails?.game_id ?? selectedPoint.game_id}
            </span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Period</span>
            <span className="flex-1 text-right">{playDetails?.period || 'N/A'}</span>
          </div>
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Description Home</span>
            <span className="flex-1 text-right">{playDetails?.event_desc_home}</span>
          </div>
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Description Away</span>
            <span className="flex-1 text-right">{playDetails?.event_desc_away}</span>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2">
        <PlayForm playDetails={playDetails} />
        {stagedChangesCount > 0 && (
          <div className="mt-6 flex w-full items-center justify-end">
            <Button
              size="sm"
              disabled={stagedChangesCount === 0 || !selectedTeamId}
              onClick={onClickApplyChanges}
            >
              Apply Changes ({stagedChangesCount})
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
}

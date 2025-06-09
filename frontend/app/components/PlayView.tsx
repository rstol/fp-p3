import { zodResolver } from '@hookform/resolvers/zod';
import { type Tag, TagInput } from 'emblor';
import { Check, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useLoaderData, useSearchParams } from 'react-router';
import { z } from 'zod';
import { BASE_URL, DefenseColor, EventType, OffenseColor } from '~/lib/const';
import { useDashboardStore } from '~/lib/stateStore';
import type { clientLoader } from '~/routes/_index';
import type { Team } from '~/types/data';
import { PlayDetailsSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from './ui/form';
import { Input } from './ui/input';
import { generateTagId } from '~/lib/utils';

interface PlayDetailState {
  videoURL?: string;
  offenseTeam?: Team;
  defenseTeam?: Team;
}

const FormSchema = z.object({
  clusters: z.array(
    z.object({
      id: z.string(),
      text: z.string(),
    }),
  ),
  tags: z.array(
    z.object({
      id: z.string(),
      text: z.string(),
    }),
  ),
  note: z.string().optional(),
});

function PlayForm() {
  const {
    movePointToCluster,
    selectedPoint,
    updatePointNote,
    updatePointTags,
    selectedCluster,
    createNewClusterWithPoint,
    createNewTagWithPoint,
    updateIsTagged,
    updateManuallyClustered,
    stageSelectedPlayClusterUpdate,
    clusters: clusterData,
    tags: tagData,
  } = useDashboardStore.getState();

  const tagOptions = tagData
    .map((t) => ({id: t.tag_label, text: t.tag_label}))
    .sort();
  const clusterOptions = clusterData
    .map((c) => ({ id: c.cluster_id, text: c.cluster_label ?? '' }))
    .sort();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { teamID } = useLoaderData<typeof clientLoader>();
  const initialCluster =
    selectedPoint?.manually_clustered && selectedCluster
      ? [{ id: selectedCluster?.cluster_id, text: selectedCluster?.cluster_label ?? '' }]
      : [];
  const initialTags =
    selectedPoint?.tags
      ? selectedPoint?.tags.map(t => ({ id: t, text: t }))
      : [];

  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      clusters: initialCluster,
      note: selectedPoint?.note ?? '',
      tags: initialTags,
    },
  });

  const [playTags, setPlayTags] = useState<Tag[]>(initialTags);
  const [clusterTags, setClusterTags] = useState<Tag[]>(initialCluster);
  const [activeTagIndex, setActiveTagIndex] = useState<number | null>(null);
  const [activeClusterIndex, setActiveClusterIndex] = useState<number | null>(null);
  const { setValue, reset } = form;

  // TODO can I fix the problem with the deleted pills here?
  useEffect(() => {
    reset({
      note: selectedPoint?.note ?? '',
      clusters: initialCluster,
      tags: initialTags,
    });
  }, [selectedPoint, selectedCluster, reset]);

  async function onSubmit(data: z.infer<typeof FormSchema>) {
    if (!selectedPoint || !selectedCluster) return;
    setIsSubmitting(true);
    const updatedCluster = data.clusters.length ? data.clusters[0] : null;
    const dataTags = data.tags.map((t) => t.text)
    const addedTags = dataTags.length ? 
      dataTags.filter(tag => !selectedPoint.tags.includes(tag))
      : null;
    const clusterPayload = updatedCluster
      ? {
          cluster_id: updatedCluster.id,
          cluster_label: updatedCluster.text,
        }
      : { ...selectedCluster }; // Always send the cluster
    console.log(initialCluster);
    await fetch(
      `${BASE_URL}/teams/${teamID}/scatterpoint/${selectedPoint?.game_id}/${selectedPoint?.event_id}`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...clusterPayload,
          note: data.note,
        }),
      },
    );
    if (updatedCluster && !clusterOptions.some((t) => t.id === updatedCluster.id)) {
      console.log({ cluster_id: updatedCluster.id, cluster_label: updatedCluster.text });
      createNewClusterWithPoint(
        { cluster_id: updatedCluster.id, cluster_label: updatedCluster.text },
        selectedPoint,
      );
      stageSelectedPlayClusterUpdate(updatedCluster?.id);
    } else if (updatedCluster && updatedCluster.id !== selectedCluster?.cluster_id) {
      movePointToCluster(selectedPoint, updatedCluster.id);
      stageSelectedPlayClusterUpdate(updatedCluster?.id);
    } else if (!initialCluster.length && updatedCluster) {
      updateManuallyClustered(selectedPoint);
      stageSelectedPlayClusterUpdate(updatedCluster?.id);
    }

    if (data.note && selectedPoint.note !== data.note) {
      updatePointNote(selectedPoint, data.note);
    }
    
    if (dataTags && selectedPoint.tags !== dataTags) {
      if (addedTags && addedTags.some((newTag) => !tagOptions.some((t => t.text === newTag))))
        addedTags.map((newTag) => createNewTagWithPoint(newTag, selectedPoint));
      updatePointTags(selectedPoint, dataTags)
      updateIsTagged(selectedPoint)
      }

  setIsSubmitting(false);
  }

  return (
    <>
      <Form {...form}>
        <form className="grid w-full items-center gap-2" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="clusters"
            render={({ field }) => (
              <FormItem className="flex flex-col items-start">
                <FormLabel className="text-sm">Manually Re-Assign Cluster</FormLabel>
                <div className="flex w-full gap-2">
                  <FormControl>
                    <TagInput
                      {...field}
                      autocompleteOptions={clusterOptions}
                      maxTags={1}
                      tags={clusterTags} // TODO
                      inlineTags
                      addTagsOnBlur
                      styleClasses={{
                        input: 'focus-visible:outline-none shadow-none w-full',
                        tag: { body: 'h-7' },
                      }}
                      generateTagId={generateTagId}
                      enableAutocomplete
                      placeholder="Select a cluster or enter a new cluster name"
                      setTags={(newTags) => {
                        setClusterTags(newTags);
                        setValue('clusters', newTags as [Tag, ...Tag[]]);
                      }}
                      activeTagIndex={activeClusterIndex}
                      setActiveTagIndex={setActiveClusterIndex}
                    />
                  </FormControl>
                  <Button
                    disabled={isSubmitting}
                    onClick={form.handleSubmit(onSubmit)}
                    size="sm"
                    className="h-9 w-10"
                  >
                    {isSubmitting ? <Loader2 className="animate-spin" /> : <Check size={6} />}
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="tags"
            render={({ field }) => (
              <FormItem className="flex flex-col items-start">
                <FormLabel className="text-sm">Tag Play</FormLabel>
                <div className="flex w-full gap-2">
                  <FormControl>
                    <TagInput
                      {...field}
                      autocompleteOptions={tagOptions}
                      tags={playTags}
                      inlineTags
                      addTagsOnBlur
                      styleClasses={{
                        input: 'focus-visible:outline-none shadow-none w-full',
                        tag: { body: 'h-7' },
                      }}
                      generateTagId={generateTagId}
                      enableAutocomplete
                      placeholder="Select or type a new tag"
                      setTags={(newTags) => {
                        setPlayTags(newTags);
                        setValue('tags', newTags as [Tag, ...Tag[]]);
                      }}
                      activeTagIndex={activeTagIndex}
                      setActiveTagIndex={setActiveTagIndex}
                    />
                  </FormControl>
                  <Button
                    disabled={isSubmitting}
                    onClick={form.handleSubmit(onSubmit)}
                    size="sm"
                    className="h-9 w-10"
                  >
                    {isSubmitting ? <Loader2 className="animate-spin" /> : <Check size={6} />}
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
                    <Input
                      disabled={isSubmitting}
                      type="text"
                      placeholder="Add a note..."
                      {...field}
                    />
                  </FormControl>
                  <Button type="submit" size="sm" className="h-9 w-10">
                    {isSubmitting ? <Loader2 className="animate-spin" /> : <Check size={6} />}
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

export default function PlayView() {
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const stagedChangesCount = useDashboardStore((state) => state.stagedChangesCount);
  const clearPendingClusterUpdates = useDashboardStore((state) => state.clearPendingClusterUpdates);
  const updateSelectedPoint = useDashboardStore((state) => state.updateSelectedPoint);
  const clearSelectedCluster = useDashboardStore((state) => state.clearSelectedCluster);
  const [playDetails, setPlayDetails] = useState<PlayDetailState | null>(null);
  const [isLoadingPlayDetails, seIsLoadingPlayDetails] = useState(false);
  const [_, setSearchParams] = useSearchParams();
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  const games = data?.games ?? [];

  useEffect(() => {
    if (!selectedPoint) return;

    seIsLoadingPlayDetails(true);
    const fetchPlayDetails = async () => {
      try {
        let publicVideoPath = `/videos/${selectedPoint.game_id}/${selectedPoint.event_id}.mp4`;

        // Try to fetch from public folder first
        const checkPublicVideo = await fetch(publicVideoPath, { method: 'HEAD' }).catch(() => ({
          ok: false,
        }));

        if (!checkPublicVideo.ok) {
          const res = await fetch(
            `${BASE_URL}/plays/${selectedPoint.game_id}/${selectedPoint.event_id}/video`,
          );
          const arrayBuffer = await res.arrayBuffer();
          const blob = new Blob([arrayBuffer], { type: 'video/mp4' });
          publicVideoPath = URL.createObjectURL(blob);
        }

        const game = games.find((game) => game.game_id === selectedPoint.game_id);
        const isHomePossession = selectedPoint.possession_team_id === game?.home_team_id;
        const offenseTeamId = isHomePossession ? game?.home_team_id : game?.visitor_team_id;
        const defenseTeamId = isHomePossession ? game?.visitor_team_id : game?.home_team_id;
        const offenseTeam = teams.find((team) => team.teamid === offenseTeamId);
        const defenseTeam = teams.find((team) => team.teamid === defenseTeamId);
        setPlayDetails({ videoURL: publicVideoPath, offenseTeam, defenseTeam });
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
          <div className="text-sm">Select a point to view the play details</div>
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
    updateSelectedPoint(null);
    clearPendingClusterUpdates();
    clearSelectedCluster();
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
            onError={(e) => console.error('Video error', e)}
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
          <div className="flex gap-4 pb-1" style={{ color: OffenseColor }}>
            <span className="shrink-0">Offense Team</span>
            <span className="flex-1 text-right">{playDetails?.offenseTeam?.name}</span>
          </div>
          <div className="flex gap-4 pb-1" style={{ color: DefenseColor }}>
            <span className="shrink-0">Defense Team</span>
            <span className="flex-1 text-right">{playDetails?.defenseTeam?.name}</span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Outcome</span>
            <span className="flex-1 text-right">
              {selectedPoint?.event_type ? `${EventType[selectedPoint.event_type]}` : 'N/A'}
            </span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Game Date</span>
            <span className="flex-1 text-right">
              {selectedPoint?.game_date ?? selectedPoint.game_date}
            </span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Quarter</span>
            <span className="flex-1 text-right">{selectedPoint?.quarter ?? 'N/A'}</span>
          </div>
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Description</span>
            <span className="flex-1 text-right">
              {selectedPoint?.event_desc_home !== 'nan' && selectedPoint?.event_desc_home}
              {selectedPoint?.event_desc_away !== 'nan' && `${selectedPoint?.event_desc_away}`}
            </span>
          </div>
          // TODO Remove this?
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Tagged</span>
            <span className="flex-1 text-right">{selectedPoint?.is_tagged ? 'Yes' : 'No'}</span>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2">
        <PlayForm />
        {stagedChangesCount > 0 && (
          <div className="mt-6 flex w-full items-center justify-end">
            <Button size="sm" disabled={stagedChangesCount === 0} onClick={onClickApplyChanges}>
              Apply Changes ({stagedChangesCount})
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
}

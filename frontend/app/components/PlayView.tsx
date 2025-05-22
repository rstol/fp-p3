import { zodResolver } from '@hookform/resolvers/zod';
import { type Tag, TagInput } from 'emblor';
import { Check } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useLoaderData, useSearchParams } from 'react-router';
import { z } from 'zod';
import { BASE_URL, DefenseColor, EventType, OffenseColor, PlayActions } from '~/lib/const';
import { useDashboardStore } from '~/lib/stateStore';
import type { clientLoader } from '~/routes/_index';
import type { PlayDetail, Team } from '~/types/data';
import { PlayDetailsSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from './ui/form';
import { Input } from './ui/input';

interface PlayDetailState extends PlayDetail {
  videoURL?: string;
  offenseTeam?: Team;
  defenseTeam?: Team;
}

const FormSchema = z.object({
  clusters: z
    .array(
      z.object({
        id: z.string(),
        text: z.string(),
      }),
    )
    .length(1),
  note: z.string().optional(),
});

export type PlayPayload = {
  data: z.infer<typeof FormSchema> & { eventId?: string; gameId?: string };
  action: PlayActions;
};

function PlayForm() {
  const data = useLoaderData<typeof clientLoader>();
  const clusterData = data?.scatterData ?? [];
  const initialTags = clusterData
    .map((c) => ({ id: c.cluster_id, text: c.cluster_label ?? '' }))
    .sort();
  const stageSelectedPlayClusterUpdate = useDashboardStore(
    (state) => state.stageSelectedPlayClusterUpdate,
  );
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  // TODO temporary transform: use data schema
  const initialCluster = selectedCluster
    ? [
        {
          id: selectedCluster.cluster_id,
          text: selectedCluster.cluster_label ?? '',
        },
      ]
    : [];

  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      clusters: initialCluster,
      note: selectedPoint?.note ?? '',
    },
  });
  const [tags, setTags] = useState<Tag[]>(initialCluster);
  const [activeTagIndex, setActiveTagIndex] = useState<number | null>(null);
  const { setValue } = form;

  async function onSubmit(data: z.infer<typeof FormSchema>) {
    const updatedCluster = data.clusters[0];
    const defaultCluster = form.formState.defaultValues?.clusters?.[0];
    if (updatedCluster.id !== defaultCluster?.id) {
      stageSelectedPlayClusterUpdate(data.clusters[0].id);
    }

    const payload = {
      cluster_id: updatedCluster.id.startsWith('new_cluster') ? null : updatedCluster.id,
      cluster_label: updatedCluster.text,
      note: data.note,
    };
    await fetch(`${BASE_URL}/scatterpoint/${selectedPoint?.game_id}/${selectedPoint?.event_id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  }

  //TODO define for to new cluster data format
  const generateTagId = () => {
    const generatedId = Math.random().toString();
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

export default function PlayView() {
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const stagedChangesCount = useDashboardStore((state) => state.stagedChangesCount);
  const clearPendingClusterUpdates = useDashboardStore((state) => state.clearPendingClusterUpdates);
  const [playDetails, setPlayDetails] = useState<PlayDetailState | null>(null);
  const [isLoadingPlayDetails, seIsLoadingPlayDetails] = useState(false);
  const [_, setSearchParams] = useSearchParams();
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  const games = data?.games ?? [];
  // TODO loader during submitting feedback & updating the clusters

  useEffect(() => {
    if (!selectedPoint) return;

    seIsLoadingPlayDetails(true);
    const fetchPlayDetails = async () => {
      try {
        let publicVideoPath = `videos/${selectedPoint.game_id}/${selectedPoint.event_id}.mp4`;

        // Try to fetch from public folder first
        const checkPublicVideo = await fetch(publicVideoPath, { method: 'HEAD' }).catch(() => ({
          ok: false,
        }));

        const resDetails = await fetch(
          `${BASE_URL}/plays/${selectedPoint.game_id}/${selectedPoint.event_id}`,
        );
        if (!resDetails.ok) throw new Error();
        const playDetails: PlayDetailState = await resDetails.json();

        if (!checkPublicVideo.ok) {
          const res = await fetch(
            `${BASE_URL}/plays/${selectedPoint.game_id}/${selectedPoint.event_id}/video`,
          );
          const arrayBuffer = await res.arrayBuffer();
          const blob = new Blob([arrayBuffer], { type: 'video/mp4' });
          publicVideoPath = URL.createObjectURL(blob);
        }

        const game = games.find((game) => game.game_id === playDetails.game_id);
        const isHomePossession = playDetails.possession_team_id === game?.home_team_id;
        const offenseTeamId = isHomePossession ? game?.home_team_id : game?.visitor_team_id;
        const defenseTeamId = isHomePossession ? game?.visitor_team_id : game?.home_team_id;
        console.log(game, offenseTeamId, defenseTeamId);
        const offenseTeam = teams.find((team) => team.teamid === offenseTeamId);
        const defenseTeam = teams.find((team) => team.teamid === defenseTeamId);
        setPlayDetails({ ...playDetails, videoURL: publicVideoPath, offenseTeam, defenseTeam });
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
  console.log(playDetails);

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
              {playDetails?.event_type ? `${EventType[playDetails.event_type]}` : 'N/A'}
            </span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Game Date</span>
            <span className="flex-1 text-right">
              {playDetails?.game_date ?? selectedPoint.game_date}
            </span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Quarter</span>
            <span className="flex-1 text-right">{playDetails?.quarter ?? 'N/A'}</span>
          </div>
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Description</span>
            <span className="flex-1 text-right">
              {playDetails?.event_desc_home !== 'nan' && playDetails?.event_desc_home}
              {playDetails?.event_desc_away !== 'nan' && `${playDetails?.event_desc_away}`}
            </span>
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

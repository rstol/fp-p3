import { zodResolver } from '@hookform/resolvers/zod';
import type {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  VisibilityState,
} from '@tanstack/react-table';
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { type Tag as TagType, TagInput } from 'emblor';
import { ArrowUpDown, ChevronDown, Edit, Eye, Loader2, MoreHorizontal, Tag } from 'lucide-react';
import * as React from 'react';
import { useForm } from 'react-hook-form';
import { useLoaderData } from 'react-router';
import { z } from 'zod';
import { Button } from '~/components/ui/button';
import { Checkbox } from '~/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '~/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '~/components/ui/dropdown-menu';
import { Input } from '~/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '~/components/ui/table';
import { BASE_URL } from '~/lib/const';
import { useDashboardStore } from '~/lib/stateStore';
import type { clientLoader } from '~/routes/_index';
import type { Point } from '~/types/data';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from './ui/form';

const EditTagFormSchema = z.object({
  clusters: z
    .array(
      z.object({
        id: z.string(),
        text: z.string(),
      }),
    )
    .length(1),
});

// Component for editing tags
function EditTagDialog({
  open,
  onOpenChange,
  selectedPlays,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedPlays: Point[];
}) {
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);
  const stageSelectedPlayClusterUpdate = useDashboardStore(
    (state) => state.stageSelectedPlayClusterUpdate,
  );
  const clusterData = useDashboardStore((state) => state.clusters);
  const tagOptions = clusterData
    .map((c) => ({ id: c.cluster_id, text: c.cluster_label ?? '' }))
    .sort();
  const allTagged = selectedPlays.every((play) => play.is_tagged);
  const initialTag =
    allTagged && selectedCluster
      ? [{ id: selectedCluster?.cluster_id, text: selectedCluster?.cluster_label ?? '' }]
      : [];
  const form = useForm<z.infer<typeof EditTagFormSchema>>({
    resolver: zodResolver(EditTagFormSchema),
    defaultValues: {
      clusters: initialTag,
    },
  });
  const { setValue } = form;
  const [tags, setTags] = React.useState<TagType[]>(initialTag);
  const [activeTagIndex, setActiveTagIndex] = React.useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const { teamID } = useLoaderData<typeof clientLoader>();

  async function onSubmit(data: z.infer<typeof EditTagFormSchema>) {
    const updatedCluster = data.clusters[0];
    const { movePointToCluster, createNewClusterWithPoint, stageSelectedPlayClusterUpdate, clusters: currentClustersInStore } = useDashboardStore.getState();
    selectedPlays.forEach(play => {
      const existingPlayCluster = currentClustersInStore.find(c => c.points.some(p => p.game_id === play.game_id && p.event_id === play.event_id));
      if (existingPlayCluster && existingPlayCluster.cluster_id === updatedCluster.id) {
        return;
      }

      if (updatedCluster.id.startsWith('new_cluster')) {
        createNewClusterWithPoint(
          { cluster_id: updatedCluster.id, cluster_label: updatedCluster.text },
          play
        );
      } else {
        movePointToCluster(play, updatedCluster.id);
      }
    });

    if (selectedPlays.length > 0) {
      stageSelectedPlayClusterUpdate(updatedCluster.id);
    }



    const payloadForBackend = {
      cluster_id: updatedCluster.id.startsWith('new_cluster') ? null : updatedCluster.id,
      cluster_label: updatedCluster.text,
    };

    const updatePayload = selectedPlays.map((play) => ({
      game_id: play.game_id,
      event_id: play.event_id,
      note: play.note,
      ...payloadForBackend,
    }));

    if (updatePayload.length > 0) {
      setIsSubmitting(true);
      try {
        const response = await fetch(`${BASE_URL}/teams/${teamID}/scatterpoints`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(updatePayload),
        });

        if (!response.ok) {
          const errorData = await response.json();
          console.error('Batch update failed:', errorData);
          throw new Error(errorData.error || 'Failed to batch update plays');
        }

        console.log('Batch update successful');
        onOpenChange(false);
      } catch (error) {
        console.error('Error during batch update:', error);
      } finally {
        setIsSubmitting(false);
      }
    }
  }

  const generateTagId = () => {
    const randomString = Math.random().toString(36).substring(2, 10); // base36, removes "0." prefix
    return `new_cluster_${randomString}`;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} id="edit-tag-form">
            <DialogHeader>
              <DialogTitle>
                {selectedPlays.length > 1
                  ? `Assign ${selectedPlays.length} plays to a cluster`
                  : 'Assign play to a cluster'}
              </DialogTitle>
              <DialogDescription>
                {selectedPlays.length > 1
                  ? 'This will update the cluster for all selected plays.'
                  : 'Update the cluster for this play.'}
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
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
                          autocompleteOptions={tagOptions}
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
                            setValue('clusters', newTags as [TagType, ...TagType[]]);
                          }}
                          activeTagIndex={activeTagIndex}
                          setActiveTagIndex={setActiveTagIndex}
                        />
                      </FormControl>
                    </div>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={isSubmitting}>
                Cancel
              </Button>
              <Button type="submit" form="edit-tag-form" disabled={isSubmitting || tags.length === 0}>
                {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Assign Cluster
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

const EditNoteFormSchema = z.object({
  note: z.string().optional(),
});

function EditNoteDialog({
  open,
  onOpenChange,
  selectedPlays,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedPlays: Point[];
}) {
  const firstNote = selectedPlays[0]?.note || '';
  const allSameNote = selectedPlays.every((play) => play.note === firstNote);
  const form = useForm<z.infer<typeof EditNoteFormSchema>>({
    resolver: zodResolver(EditNoteFormSchema),
    defaultValues: {
      note: allSameNote ? firstNote : '',
    },
  });

  async function onSubmit(data: z.infer<typeof EditNoteFormSchema>) {
    console.log('submit', data);
    // TODO bulk endpoint for all points
    const updatePlayIds = selectedPlays.map((play) => ({
      game_id: play.game_id,
      event_id: play.event_id,
    }));
    // await fetch(`${BASE_URL}/teams/${teamID}/scatterpoint/${selectedPoint?.game_id}/${selectedPoint?.event_id}`, {
    //   method: 'PUT',
    //   headers: {
    //     'Content-Type': 'application/json',
    //   },
    //   body: JSON.stringify({play_ids: updatePlayIds, note: data.note}),
    // });
    //TODO handle state updates like in PlayView.PlayForm or do an "apply-all" button submission
    onOpenChange(false);
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <DialogHeader>
              <DialogTitle>Edit play note</DialogTitle>
              <DialogDescription>Update the note for this play.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <FormField
                control={form.control}
                name="note"
                render={({ field }) => (
                  <FormItem className="flex flex-col items-start">
                    <FormLabel className="text-sm">Play Note</FormLabel>
                    <FormControl>
                      <Input type="text" placeholder="Add a note..." {...field} />
                    </FormControl>

                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button type="submit">Save</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

export function PlaysTable({ title, data }: { title: string; data: Point[] }) {
  const [sorting, setSorting] = React.useState<SortingState>([
    {
      id: 'similarity_distance',
      desc: true,
    },
  ]);
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = React.useState({});
  const selectedCluster = useDashboardStore((state) => state.selectedCluster);
  const loaderData = useLoaderData<typeof clientLoader>();
  const { games, teams } = loaderData;

  const gameMap = new Map(games?.map((game) => [game.game_id, game]));
  const teamMap = new Map(teams.map((team) => [team.teamid, team.name]));
  const enhancedData =
    React.useMemo(
      () =>
        data.map((point) => {
          const game = gameMap.get(point.game_id);
          const visitorTeamName = game ? teamMap.get(game.visitor_team_id) : undefined;

          return {
            ...point,
            videoURL: `/videos/${point.game_id}/${point.event_id}.mp4`,
            visitorTeamName,
          };
        }),
      [data],
    ) ?? [];

  // Dialog states
  const [tagDialogOpen, setTagDialogOpen] = React.useState(false);
  const [noteDialogOpen, setNoteDialogOpen] = React.useState(false);
  const [selectedPlays, setSelectedPlays] = React.useState<Point[]>([]);
  const updateSelectedPoint = useDashboardStore((state) => state.updateSelectedPoint);
  // TODO Loading indicator

  // Action handlers
  function handleEditTag(play: Point) {
    setSelectedPlays([play]);
    setTagDialogOpen(true);
  }

  function handleEditNote(play: Point) {
    setSelectedPlays([play]);
    setNoteDialogOpen(true);
  }

  function handleViewPreview(play: Point) {
    updateSelectedPoint(play);
  }

  function handleTagSelected() {
    const selectedRows = table.getFilteredSelectedRowModel().rows;
    const plays = selectedRows.map((row) => row.original);
    setSelectedPlays(plays);
    setTagDialogOpen(true);
  }

  // Define columns
  const columns: ColumnDef<Point>[] = [
    {
      id: 'select',
      header: ({ table }) => (
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && 'indeterminate')
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
          aria-label="Select all"
        />
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => row.toggleSelected(!!value)}
          aria-label="Select row"
        />
      ),
      enableSorting: false,
      enableHiding: false,
    },
    {
      accessorKey: 'similarity_distance',
      header: ({ column }) => (
        <Button
          variant="ghost"
          className="px-0.5!"
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
        >
          Similarity Score
          <ArrowUpDown size={4} />
        </Button>
      ),
      cell: ({ row }) => (
        <div>{(100 * Number(row.getValue('similarity_distance'))).toFixed(2)}</div>
      ),
    },
    {
      accessorKey: 'game_date',
      header: ({ column }) => (
        <Button
          variant="ghost"
          className="px-0.5!"
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
        >
          Game Date
          <ArrowUpDown size={4} />
        </Button>
      ),
      cell: ({ row }) => {
        const date = new Date(row.getValue('game_date'));
        return <div>{date.toLocaleDateString()}</div>;
      },
    },
    {
      accessorKey: 'quarter',
      header: ({ column }) => (
        <Button
          variant="ghost"
          className="px-0.5!"
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
        >
          Quarter
          <ArrowUpDown size={4} />
        </Button>
      ),
      cell: ({ row }) => <div>Q{row.getValue('quarter')}</div>,
    },
    {
      accessorKey: 'visitorTeamName',
      header: 'Opponent Team',
      cell: ({ row }) => <div>{row.getValue('visitorTeamName')}</div>,
    },
    {
      accessorKey: 'note',
      header: 'Play Note',
      cell: ({ row }) => (
        <div className="w-[150px] whitespace-break-spaces" title={row.getValue('note')}>
          {row.getValue('note')}
        </div>
      ),
    },
    {
      accessorKey: 'videoURL',
      header: 'Play Video',
      cell: ({ row }) => (
        <div className="max-w-[250px] min-w-[200px] truncate" title={row.getValue('videoURL')}>
          <video
            key={row.getValue('videoURL')} // Force remount component on change
            controls
            disablePictureInPicture
            disableRemotePlayback
            muted
          >
            <source src={row.getValue('videoURL')} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      ),
    },
    {
      id: 'actions',
      enableHiding: false,
      cell: ({ row }) => {
        const play = row.original;

        return (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-8 w-8 p-0">
                <span className="sr-only">Open menu</span>
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Actions</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => handleEditTag(play)}>
                <Tag className="mr-1 h-4 w-4" />
                Assign different cluster
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleEditNote(play)}>
                <Edit className="mr-1 h-4 w-4" />
                Edit play note
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleViewPreview(play)}>
                <Eye className="mr-1 h-4 w-4" />
                Preview Play in Plot
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        );
      },
    },
  ];

  const table = useReactTable({
    data: enhancedData,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
    },
  });

  if (!selectedCluster) return null;

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold">{title}</h2>
      <EditTagDialog
        open={tagDialogOpen}
        onOpenChange={setTagDialogOpen}
        selectedPlays={selectedPlays}
      />
      <EditNoteDialog
        open={noteDialogOpen}
        onOpenChange={setNoteDialogOpen}
        selectedPlays={selectedPlays}
      />
      <div className="flex items-center justify-between py-4">
        <div className="flex gap-2">
          <Input
            placeholder="Filter by note..."
            value={(table.getColumn('note')?.getFilterValue() as string) ?? ''}
            onChange={(event) => table.getColumn('note')?.setFilterValue(event.target.value)}
            className="max-w-sm"
          />
          <Input
            placeholder="Filter by team..."
            value={(table.getColumn('visitorTeamName')?.getFilterValue() as string) ?? ''}
            onChange={(event) =>
              table.getColumn('visitorTeamName')?.setFilterValue(event.target.value)
            }
            className="max-w-sm"
          />
        </div>
        <div className="flex gap-2">
          {table.getFilteredSelectedRowModel().rows.length > 0 ? (
            <Button
              variant="secondary"
              onClick={handleTagSelected}
              className="flex items-center gap-1"
            >
              <Tag className="h-4 w-4" />
              Assign selected plays to a cluster ({table.getFilteredSelectedRowModel().rows.length})
            </Button>
          ) : null}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="ml-auto">
                Columns <ChevronDown className="ml-2 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {table
                .getAllColumns()
                .filter((column) => column.getCanHide())
                .map((column) => {
                  return (
                    <DropdownMenuCheckboxItem
                      key={column.id}
                      className="capitalize"
                      checked={column.getIsVisible()}
                      onCheckedChange={(value) => column.toggleVisibility(!!value)}
                    >
                      {column.id}
                    </DropdownMenuCheckboxItem>
                  );
                })}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(header.column.columnDef.header, header.getContext())}
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} data-state={row.getIsSelected() ? 'selected' : undefined}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center">
                  No point selected.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-end space-x-2 py-4">
        <div className="text-muted-foreground flex-1 text-sm">
          {table.getFilteredSelectedRowModel().rows.length} of{' '}
          {table.getFilteredRowModel().rows.length} row(s) selected.
        </div>
        <div className="space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}

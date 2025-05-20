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
import { ArrowUpDown, ChevronDown, Edit, Eye, MoreHorizontal, Tag } from 'lucide-react';
import * as React from 'react';
import { useForm } from 'react-hook-form';
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
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from './ui/form';
import { useDashboardStore } from '~/lib/stateStore';

// Define the Play type
export type Play = {
  id: string;
  clusterId: number;
  gameDate: Date;
  quarter: number;
  gameClock: string;
  homeTeam: string;
  awayTeam: string;
  playNote: string;
  playVideoUrl: string;
};

// Sample data for demonstration
const data: Play[] = [
  {
    id: 'play1',
    clusterId: 1,
    gameDate: new Date('2025-04-01'),
    quarter: 1,
    gameClock: '10:45',
    homeTeam: 'Lakers',
    awayTeam: 'Celtics',
    playNote: '',
    playVideoUrl: 'videos/0021500615/95.mp4',
  },
  {
    id: 'play2',
    clusterId: 1,
    gameDate: new Date('2025-04-01'),
    quarter: 2,
    gameClock: '5:22',
    homeTeam: 'Lakers',
    awayTeam: 'Warriors',
    playNote: 'Curry isolation floater',
    playVideoUrl: 'videos/0021500615/95.mp4',
  },
  {
    id: 'play3',
    clusterId: 1,
    gameDate: new Date('2025-04-03'),
    quarter: 4,
    gameClock: '2:15',
    homeTeam: 'Lakers',
    awayTeam: 'Bucks',
    playNote: 'Middleton isolation jump-ball two-pointer',
    playVideoUrl: 'videos/0021500615/95.mp4',
  },
  {
    id: 'play4',
    clusterId: 1,
    gameDate: new Date('2025-04-05'),
    quarter: 3,
    gameClock: '7:33',
    homeTeam: 'Lakers',
    awayTeam: 'Nuggets',
    playNote: 'Jokić Isolation against Davis',
    playVideoUrl: 'videos/0021500615/95.mp4',
  },
  {
    id: 'play5',
    clusterId: 1,
    gameDate: new Date('2025-04-05'),
    quarter: 4,
    gameClock: '0:45',
    homeTeam: 'Lakers',
    awayTeam: 'Suns',
    playNote: 'Durant three-pointer',
    playVideoUrl: 'videos/0021500615/95.mp4',
  },
];

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
  selectedPlays: any[];
}) {
  const stageSelectedPlayClusterUpdate = useDashboardStore(
    (state) => state.stageSelectedPlayClusterUpdate,
  );
  const initialCluster = { id: '', text: '' };
  const initialTags = [initialCluster];
  const form = useForm<z.infer<typeof EditTagFormSchema>>({
    resolver: zodResolver(EditTagFormSchema),
    defaultValues: {
      clusters: [initialCluster],
    },
  });
  const { setValue } = form;
  const [tags, setTags] = React.useState<TagType[]>([initialCluster]);
  const [activeTagIndex, setActiveTagIndex] = React.useState<number | null>(null);
  function onSubmit(data: z.infer<typeof EditTagFormSchema>) {
    const updatedCluster = data.clusters[0];
    const defaultCluster = form.formState.defaultValues?.clusters?.[0];
    console.log('submit', data);
    if (updatedCluster.text !== defaultCluster?.text && updatedCluster.id !== defaultCluster?.id) {
      // Submit
      stageSelectedPlayClusterUpdate(updatedCluster.id);
    }
    onOpenChange(false);
  }

  const generateTagId = () => {
    const generatedId = Math.random() * 16;
    return `new_cluster_${generatedId}`;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
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
              <Button onClick={() => onOpenChange(false)} variant="outline">
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

const EditNoteFormSchema = z.object({
  note: z.string().optional(),
});

function EditNoteDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const form = useForm<z.infer<typeof EditNoteFormSchema>>({
    resolver: zodResolver(EditNoteFormSchema),
    defaultValues: {
      note: '', // playDetails.note TODO
    },
  });

  function onSubmit(data: z.infer<typeof EditNoteFormSchema>) {
    console.log('submit', data);
    if (form.formState.defaultValues?.note !== form.formState.defaultValues?.note) {
      // TODO submit
    }
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

export function PlaysTable() {
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = React.useState({});
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  // Dialog states
  const [tagDialogOpen, setTagDialogOpen] = React.useState(false);
  const [noteDialogOpen, setNoteDialogOpen] = React.useState(false);
  const [selectedPlay, setSelectedPlay] = React.useState<Play | null>(null);
  const [selectedPlays, setSelectedPlays] = React.useState<Play[]>([]);
  const [editTag, setEditTag] = React.useState('');
  const [editNote, setEditNote] = React.useState('');

  // TODO use scatter data for this list or another endpoint
  // Loading // Hide if no Play is selected!

  // Action handlers
  function handleEditTag(play: Play) {
    setSelectedPlay(play);
    setEditTag(String(play.clusterId));
    setTagDialogOpen(true);
  }

  function handleEditNote(play: Play) {
    setSelectedPlay(play);
    setEditNote(play.playNote);
    setNoteDialogOpen(true);
  }

  function handleViewPreview(playId: string) {
    console.log(`Preview Play in Plot for play: ${playId}`);
    // Implement your preview logic here
  }

  function handleTagSelected() {
    const selectedRows = table.getFilteredSelectedRowModel().rows;
    const plays = selectedRows.map((row) => row.original);
    setSelectedPlays(plays);
    // Use the tag from the first play as initial value, or empty string if plays have different tags
    const firstTag = plays[0]?.clusterId || '';
    const allSameTag = plays.every((play) => play.clusterId === firstTag);
    setEditTag(String(allSameTag ? firstTag : ''));
    setTagDialogOpen(true);
  }

  // Define columns
  const columns: ColumnDef<Play>[] = [
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
      accessorKey: 'similarityScore',
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
      cell: ({ row }) => <div>{row.getValue('similarityScore')}</div>,
    },
    // {
    //   accessorKey: 'clusterId',
    //   header: ({ column }) => (
    //     <Button
    //       variant="ghost"
    //       className="px-0.5!"
    //       onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
    //     >
    //       Cluster
    //       <ArrowUpDown size={4} />
    //     </Button>
    //   ),
    //   cell: ({ row }) => <div>{row.getValue('clusterId')}</div>,
    // },
    {
      accessorKey: 'gameDate',
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
        const date = row.getValue('gameDate') as Date;
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
      accessorKey: 'gameClock',
      header: ({ column }) => (
        <Button
          variant="ghost"
          className="px-0.5!"
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
        >
          Game Clock
          <ArrowUpDown size={4} />
        </Button>
      ),
      cell: ({ row }) => <div>{row.getValue('gameClock')}</div>,
    },
    {
      accessorKey: 'awayTeam',
      header: 'Opponent Team',
      cell: ({ row }) => <div>{row.getValue('awayTeam')}</div>,
    },
    {
      accessorKey: 'playNote',
      header: 'Play Note',
      cell: ({ row }) => (
        <div className="w-[150px] whitespace-break-spaces" title={row.getValue('playNote')}>
          {row.getValue('playNote')}
        </div>
      ),
    },
    {
      accessorKey: 'playVideoUrl',
      header: 'Play Video',
      cell: ({ row }) => (
        <div className="max-w-[250px] min-w-[200px] truncate" title={row.getValue('playVideoUrl')}>
          <video
            key={row.getValue('playVideoUrl')} // Force remount component on change
            controls
            disablePictureInPicture
            disableRemotePlayback
            muted
          >
            <source src={row.getValue('playVideoUrl')} type="video/mp4" />
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
              <DropdownMenuItem onClick={() => handleViewPreview(play.id)}>
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
    data,
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

  const title = `Similar plays in cluster ${selectedPoint?.cluster ?? ''}`;

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold">{title}</h2>
      <EditTagDialog
        open={tagDialogOpen}
        onOpenChange={setTagDialogOpen}
        selectedPlays={selectedPlays}
      />
      <EditNoteDialog open={noteDialogOpen} onOpenChange={setNoteDialogOpen} />
      <div className="flex items-center justify-between py-4">
        <div className="flex gap-2">
          <Input
            placeholder="Filter by note..."
            value={(table.getColumn('playNote')?.getFilterValue() as string) ?? ''}
            onChange={(event) => table.getColumn('playNote')?.setFilterValue(event.target.value)}
            className="max-w-sm"
          />
          <Input
            placeholder="Filter by team..."
            value={(table.getColumn('awayTeam')?.getFilterValue() as string) ?? ''}
            onChange={(event) => table.getColumn('awayTeam')?.setFilterValue(event.target.value)}
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
                  No results.
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

import * as React from 'react';
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
import { ArrowUpDown, ChevronDown, MoreHorizontal, Edit, Eye, Tag, Save, X } from 'lucide-react';
import { Button } from '~/components/ui/button';
import { Checkbox } from '~/components/ui/checkbox';
import {
  Dialog,
  DialogClose,
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
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '~/components/ui/dropdown-menu';
import { Input } from '~/components/ui/input';
import { Label } from '~/components/ui/label';
import { Textarea } from '~/components/ui/textarea';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '~/components/ui/table';

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

// Component for editing tags
function EditTagDialog({
  open,
  onOpenChange,
  selectedPlays,
  editTag,
  setEditTag,
  saveTag,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedPlays: any[];
  editTag: string;
  setEditTag: (value: string) => void;
  saveTag: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {selectedPlays.length > 1
              ? `Edit tag for ${selectedPlays.length} plays`
              : 'Edit play tag'}
          </DialogTitle>
          <DialogDescription>
            {selectedPlays.length > 1
              ? 'This will update the tag for all selected plays.'
              : 'Update the tag for this play.'}
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="tag">Tag</Label>
            <Input
              id="tag"
              value={editTag}
              onChange={(e) => setEditTag(e.target.value)}
              placeholder="Enter play tag"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={saveTag}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Component for editing notes
function EditNoteDialog({
  open,
  onOpenChange,
  editNote,
  setEditNote,
  saveNote,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editNote: string;
  setEditNote: (value: string) => void;
  saveNote: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit play note</DialogTitle>
          <DialogDescription>Update the note for this play.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="note">Note</Label>
            <Textarea
              id="note"
              value={editNote}
              onChange={(e) => setEditNote(e.target.value)}
              placeholder="Enter play note"
              rows={4}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={saveNote}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function PlaysTable({ title }: { title: string }) {
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = React.useState({});

  // Dialog states
  const [tagDialogOpen, setTagDialogOpen] = React.useState(false);
  const [noteDialogOpen, setNoteDialogOpen] = React.useState(false);
  const [selectedPlay, setSelectedPlay] = React.useState<Play | null>(null);
  const [selectedPlays, setSelectedPlays] = React.useState<Play[]>([]);
  const [editTag, setEditTag] = React.useState('');
  const [editNote, setEditNote] = React.useState('');

  // Action handlers
  function handleEditTag(play: Play) {
    setSelectedPlay(play);
    setEditTag(play.clusterId);
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
    setEditTag(allSameTag ? firstTag : '');
    setTagDialogOpen(true);
  }

  function saveTag() {
    if (selectedPlay) {
      console.log(`Saving tag "${editTag}" for play: ${selectedPlay.id}`);
      // Update tag logic for single play here
    } else if (selectedPlays.length > 0) {
      console.log(
        `Saving tag "${editTag}" for plays: ${selectedPlays.map((p) => p.id).join(', ')}`,
      );
      // Update tag logic for multiple plays here
    }
    setTagDialogOpen(false);
  }

  function saveNote() {
    if (selectedPlay) {
      console.log(`Saving note "${editNote}" for play: ${selectedPlay.id}`);
      // Update note logic here
    }
    setNoteDialogOpen(false);
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
                <Edit className="mr-2 h-4 w-4" />
                Edit Cluster
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleEditNote(play)}>
                <Edit className="mr-2 h-4 w-4" />
                Edit play note
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleViewPreview(play.id)}>
                <Eye className="mr-2 h-4 w-4" />
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

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold">{title}</h2>
      <EditTagDialog
        open={tagDialogOpen}
        onOpenChange={setTagDialogOpen}
        selectedPlays={selectedPlays}
        editTag={editTag}
        setEditTag={setEditTag}
        saveTag={saveTag}
      />
      <EditNoteDialog
        open={noteDialogOpen}
        onOpenChange={setNoteDialogOpen}
        editNote={editNote}
        setEditNote={setEditNote}
        saveNote={saveNote}
      />
      <div className="flex items-center justify-between py-4">
        <div className="flex gap-2">
          <Input
            placeholder="Filter by tag..."
            value={(table.getColumn('playTag')?.getFilterValue() as string) ?? ''}
            onChange={(event) => table.getColumn('playTag')?.setFilterValue(event.target.value)}
            className="max-w-sm"
          />
          <Input
            placeholder="Filter by team..."
            value={(table.getColumn('homeTeam')?.getFilterValue() as string) ?? ''}
            onChange={(event) => table.getColumn('homeTeam')?.setFilterValue(event.target.value)}
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
              Tag Selected ({table.getFilteredSelectedRowModel().rows.length})
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

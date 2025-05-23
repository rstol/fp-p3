import { Check, ChevronDown, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useLoaderData, useNavigation, useSearchParams } from 'react-router';
import { Button } from '~/components/ui/button';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '~/components/ui/command';
import { Label } from '~/components/ui/label';
import { Popover, PopoverContent, PopoverTrigger } from '~/components/ui/popover';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '~/components/ui/select';
import { GameFilter, TeamIDs } from '~/lib/const';
import { cn } from '~/lib/utils';
import type { clientLoader } from '~/routes/_index';

const allGameFilters = [
  { value: GameFilter.LAST1, label: 'Last Game' },
  { value: GameFilter.LAST2, label: 'Last 2 Games' },
  { value: GameFilter.LAST3, label: 'Last 3 Games' },
  { value: GameFilter.LAST4, label: 'Last 4 Games' },
  { value: GameFilter.LAST5, label: 'Last 5 Games' },
];

export default function Filters({ teamID }: { teamID: string | null }) {
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  const [open, setOpen] = useState(false);
  const [_, setSearchParams] = useSearchParams();
  const navigation = useNavigation();
  const isNavigating = Boolean(navigation.location);
  const filteredTeams = teams.filter((team) => TeamIDs.includes(team.teamid));

  const commandItems = filteredTeams.map((team) => ({
    value: String(team.teamid),
    label: team.name,
    abbreviation: team.abbreviation,
  }));

  const timeframe = data.timeframe ?? GameFilter.LAST3;
  const totalGames = data.totalGames;

  return (
    <div className="absolute top-2 left-2 z-10 flex gap-3">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <div className="grid gap-1.5">
            <Button
              role="combobox"
              aria-expanded={open}
              className="items-center gap-1"
              disabled={isNavigating}
            >
              <Label className="text-xs">Team:</Label>
              {isNavigating ? (
                <Loader2 className="animate-spin" />
              ) : teamID ? (
                commandItems.find((team) => team.value === teamID)?.label
              ) : (
                'Select Team'
              )}
              <ChevronDown size={7} className="shrink-0" />
            </Button>
          </div>
        </PopoverTrigger>
        <PopoverContent className="w-[250px] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search team" />
            <CommandList>
              <CommandEmpty>No teams found.</CommandEmpty>
              <CommandGroup>
                {commandItems.map((team) => (
                  <CommandItem
                    key={team.value}
                    value={team.value}
                    keywords={[team.label]}
                    onSelect={(currentTeamId) => {
                      setOpen(false);
                      setSearchParams((prev) => {
                        prev.set('teamid', currentTeamId);
                        return prev;
                      });
                    }}
                  >
                    <Check
                      className={cn(
                        'mr-1 h-4 w-4',
                        teamID === team.value ? 'opacity-100' : 'opacity-0',
                      )}
                    />
                    {team.label}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      <div className="grid gap-1">
        <Select
          value={String(timeframe)}
          onValueChange={(value: string) => {
            setSearchParams((prev) => {
              prev.set('timeframe', value);
              return prev;
            });
          }}
          disabled={isNavigating}
        >
          <SelectTrigger className="gap-1.5 border border-gray-400 bg-white">
            <Label htmlFor="timeframe" className="text-xs">
              Timeframe:
            </Label>
            {isNavigating ? (
              <Loader2 className="animate-spin" />
            ) : (
              <SelectValue placeholder="Timeframe" />
            )}
          </SelectTrigger>
          <SelectContent id="timeframe">
            <SelectGroup>
              {allGameFilters
                .filter((filter) => filter.value <= totalGames)
                .map((filter) => (
                  <SelectItem key={filter.value} value={String(filter.value)}>
                    {filter.label}
                  </SelectItem>
                ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

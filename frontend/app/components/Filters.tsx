import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from '~/components/ui/select';
import { Check, ChevronDown, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { useLoaderData, useNavigate, useNavigation } from 'react-router';
import { Button } from '~/components/ui/button';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '~/components/ui/command';
import { Popover, PopoverContent, PopoverTrigger } from '~/components/ui/popover';
import { GameFilter, useDashboardStore } from '~/lib/stateStore';
import { cn } from '~/lib/utils';
import type { clientLoader } from '~/routes/_index';
import { Label } from '~/components/ui/label';

const lastGamesFilter = [
  { value: GameFilter.LAST5, label: 'Last 5 Games' },
  { value: GameFilter.LAST4, label: 'Last 4 Games' },
  { value: GameFilter.LAST3, label: 'Last 3 Games' },
];

export default function Filters({ teamID }: { teamID: string | null }) {
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  const [open, setOpen] = useState(false);
  const gameFilter = useDashboardStore((state) => state.gameFilter);
  const updateGameFilter = useDashboardStore((state) => state.updateGameFilter);
  let navigate = useNavigate();
  const navigation = useNavigation();
  const isNavigating = Boolean(navigation.location);

  const commandItems = teams.map((team) => ({
    value: String(team.teamid),
    label: team.name,
    abbreviation: team.abbreviation,
  }));

  return (
    <div className="mt-5 flex gap-4">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <div className="grid gap-1.5">
            <Label>All Teams</Label>
            <Button
              role="combobox"
              aria-expanded={open}
              className="items-center"
              disabled={isNavigating}
            >
              {isNavigating && <Loader2 className="animate-spin" />}
              {teamID ? commandItems.find((team) => team.value === teamID)?.label : 'Select Team'}
              <ChevronDown size={7} className="shrink-0" />
            </Button>
          </div>
        </PopoverTrigger>
        <PopoverContent className="w-[250px] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search team" />
            <CommandList>
              <CommandEmpty>No teans found.</CommandEmpty>
              <CommandGroup>
                {commandItems.map((team) => (
                  <CommandItem
                    key={team.value}
                    value={team.value}
                    keywords={[team.label]}
                    onSelect={(currentTeamId) => {
                      setOpen(false);
                      navigate(currentTeamId === teamID ? '/' : `/?teamid=${currentTeamId}`);
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
      <div className="grid gap-1.5">
        <Label htmlFor="timeframe">Timeframe</Label>
        <Select value={gameFilter} onValueChange={updateGameFilter}>
          <SelectTrigger className="">
            <SelectValue placeholder="Timeframe" />
          </SelectTrigger>
          <SelectContent id="timeframe">
            <SelectGroup>
              {lastGamesFilter.map((filter) => (
                <SelectItem key={filter.value} value={filter.value}>
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

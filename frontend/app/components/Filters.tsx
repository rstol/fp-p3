import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from '~/components/ui/select';
import { Check, ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { useLoaderData } from 'react-router';
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
import type { clientLoader } from '~/routes/home';

const lastGamesFilter = [
  { value: GameFilter.LAST5, label: 'Last 5 Games' },
  { value: GameFilter.LAST4, label: 'Last 4 Games' },
  { value: GameFilter.LAST3, label: 'Last 3 Games' },
];

export default function Filters() {
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  const [open, setOpen] = useState(false);
  const homeTeamId = useDashboardStore((state) => state.homeTeamId);
  const updateHomeTeamId = useDashboardStore((state) => state.updateHomeTeamId);
  const gameFilter = useDashboardStore((state) => state.gameFilter);
  const updateGameFilter = useDashboardStore((state) => state.updateGameFilter);

  const commandItems = teams.map((team) => ({
    value: String(team.teamid),
    label: `${team.name} (${team.abbreviation})`,
    abbreviation: team.abbreviation,
  }));

  return (
    <div className="mt-5 flex gap-4">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            role="combobox"
            aria-expanded={open}
            className="h-auto flex-col items-center gap-0 p-0 text-2xl"
          >
            {homeTeamId
              ? commandItems.find((team) => team.value === homeTeamId)?.abbreviation
              : 'Select team'}
            <ChevronDown size={28} className="shrink-0" />
          </Button>
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
                      updateHomeTeamId(currentTeamId === homeTeamId ? '' : currentTeamId);
                      setOpen(false);
                    }}
                  >
                    <Check
                      className={cn(
                        'mr-1 h-4 w-4',
                        homeTeamId === team.value ? 'opacity-100' : 'opacity-0',
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
      <Select value={gameFilter} onValueChange={updateGameFilter}>
        <SelectTrigger className="">
          <SelectValue placeholder="Last N Games" />
        </SelectTrigger>
        <SelectContent>
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
  );
}

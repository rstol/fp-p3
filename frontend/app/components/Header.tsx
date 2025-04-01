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
import { useDashboardStore } from '~/lib/stateStore';
import { cn } from '~/lib/utils';
import type { clientLoader } from '~/routes/home';

export default function Header() {
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  const [open, setOpen] = useState(false);
  const homeTeamId = useDashboardStore((state) => state.homeTeamId);
  const updateHomeTeamId = useDashboardStore((state) => state.updateHomeTeamId);

  const commandItems = teams.map((team) => ({
    value: String(team.teamid),
    label: `${team.name} (${team.abbreviation})`,
    abbreviation: team.abbreviation,
  }));

  return (
    <div>
      <h1 className="text-lg font-medium">Basketball Play Analyzer</h1>
      <div className="mt-5">
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              role="combobox"
              aria-expanded={open}
              className="h-auto flex-col items-center gap-0 text-2xl"
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
      </div>
    </div>
  );
}

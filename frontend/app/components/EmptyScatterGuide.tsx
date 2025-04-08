import { Link, NavLink, useLoaderData } from 'react-router';
import { clientLoader } from '~/routes/_index';
import type { Team } from '~/types/data';
import { Button } from './ui/button';

export default function EmptyScatterGuide() {
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];
  return (
    <div className="flex h-full w-full flex-col items-center justify-center p-6">
      <h2 className="mb-2 text-xl font-medium">Select a Team to View Play Data</h2>
      <p className="mb-8 max-w-md text-center text-gray-500">
        View basketball play positions clustered by similarity. Compare patterns across games and
        identify strategic trends.
      </p>

      <div className="flex flex-wrap justify-center">
        {teams.map((team) => (
          <TeamThumbnail key={team.teamid} team={team} isSelected={false} />
        ))}
      </div>
    </div>
  );
}

function TeamThumbnail({ team, isSelected }: { team: Team; isSelected: boolean }) {
  return (
    <Link to={{ pathname: '/', search: `?teamid=${team.teamid}` }}>
      <Button
        variant={isSelected ? 'default' : 'outline'}
        className={`m-2 flex h-auto w-40 flex-col items-center justify-center gap-0 px-2! transition-all ${
          isSelected ? 'ring-2 ring-offset-2' : ''
        }`}
        style={{
          backgroundColor: isSelected ? 'red-300' : 'transparent',
          color: isSelected ? 'white' : 'inherit',
          borderColor: 'red-300',
        }}
      >
        <span className="text-xl font-bold">{team.abbreviation}</span>
        <span className="max-w-full truncate text-center text-xs">{team.name}</span>
      </Button>
    </Link>
  );
}

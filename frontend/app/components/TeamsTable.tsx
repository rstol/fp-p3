import { useLoaderData } from 'react-router';
import type { clientLoader } from '~/routes/home';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table';

export default function TeamsTable() {
  const data = useLoaderData<typeof clientLoader>();
  const teams = data?.teams ?? [];

  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold">Teams</h2>
      <Table>
        <TableCaption>A list all teams in db.</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[100px]">Abbreviation</TableHead>
            <TableHead>Name</TableHead>
            <TableHead>Players</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {teams.map((team) => (
            <TableRow key={team.teamid}>
              <TableCell>{team.abbreviation}</TableCell>
              <TableCell>{team.name}</TableCell>
              {team.players.map((player) => (
                <TableCell className="font-medium" key={player.playerid}>
                  {player.firstname} {player.lastname}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

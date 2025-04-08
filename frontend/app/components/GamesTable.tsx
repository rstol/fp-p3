import { useLoaderData } from 'react-router';
import type { clientLoader } from '~/routes/_index';
import {
  TableCaption,
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from './ui/table';

export default function GamesTable() {
  const data = useLoaderData<typeof clientLoader>();
  const games = data?.games ?? [];

  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold">Games</h2>

      <Table>
        <TableCaption>A list of all games in the database.</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[150px]">Game ID</TableHead>
            <TableHead>Game Date</TableHead>
            <TableHead>Home Team ID</TableHead>
            <TableHead>Visitor Team ID</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {games.map((game) => (
            <TableRow key={game.game_id}>
              <TableCell>{game.game_id}</TableCell>
              <TableCell>{game.game_date}</TableCell>
              <TableCell>{game.home_team_id}</TableCell>
              <TableCell>{game.visitor_team_id}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

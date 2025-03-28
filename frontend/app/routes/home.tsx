import { Card, CardContent, CardHeader, CardTitle } from '~/components/ui/card';
import type { Route } from './+types/home';
import { Button } from '~/components/ui/button';
import { useState } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '~/components/ui/select';
import { Slider } from '~/components/ui/slider';

export function meta({}: Route.MetaArgs) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}

const clusters = [
  {
    id: 1,
    name: 'Pick and Roll',
    color: '#fee2e2',
    borderColor: '#ef4444',
    plays: [
      { id: 1, x: 160, y: 140 },
      { id: 2, x: 190, y: 150 },
      { id: 3, x: 200, y: 180 },
      { id: 4, x: 170, y: 175 },
      { id: 5, x: 150, y: 160 },
    ],
  },
  {
    id: 2,
    name: 'Isolation',
    color: '#dbeafe',
    borderColor: '#3b82f6',
    plays: [
      { id: 6, x: 350, y: 160 },
      { id: 7, x: 375, y: 190 },
      { id: 8, x: 395, y: 175 },
      { id: 9, x: 360, y: 195 },
    ],
  },
  {
    id: 3,
    name: 'Fast Break',
    color: '#dcfce7',
    borderColor: '#16a34a',
    plays: [
      { id: 10, x: 270, y: 260 },
      { id: 11, x: 305, y: 275 },
      { id: 12, x: 290, y: 290 },
      { id: 13, x: 315, y: 305 },
      { id: 14, x: 275, y: 295 },
    ],
  },
];

// Court player positions for selected play
const courtPlayers = [
  { id: 1, x: 40, y: 40, team: 'offense' },
  { id: 2, x: 60, y: 70, team: 'offense' },
  { id: 3, x: 82, y: 30, team: 'offense' },
  { id: 4, x: 110, y: 40, team: 'offense' },
  { id: 5, x: 100, y: 80, team: 'offense' },
  { id: 6, x: 45, y: 30, team: 'defense' },
  { id: 7, x: 75, y: 60, team: 'defense' },
  { id: 8, x: 92, y: 25, team: 'defense' },
  { id: 9, x: 120, y: 30, team: 'defense' },
  { id: 10, x: 105, y: 70, team: 'defense' },
];

export default function Home() {
  // State
  const [selectedPlay, setSelectedPlay] = useState<number | null>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState([50]);
  const [clusterMethod, setClusterMethod] = useState('t-SNE');
  const [featureImportance, setFeatureImportance] =
    useState('Player Positions');
  const [selectedTeams, setSelectedTeams] = useState(['Warriors']);
  // Toggle team selection
  const toggleTeam = (team: string) => {
    if (selectedTeams.includes(team)) {
      setSelectedTeams(selectedTeams.filter((t) => t !== team));
    } else {
      setSelectedTeams([...selectedTeams, team]);
    }
  };
  return (
    <div className="max-w-7xl mx-auto py-6">
      <CardHeader className="pb-2">
        <CardTitle className="text-center text-xl pb-2">
          Basketball Play Tactics Tool
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Teams Navigation */}
        <div className="bg-blue-900 p-3 flex items-center space-x-2">
          <span className="text-white font-medium">Teams</span>
          <Button
            variant={
              selectedTeams.includes('Warriors') ? 'default' : 'secondary'
            }
            className={
              selectedTeams.includes('Warriors') ? 'bg-blue-600' : 'bg-gray-600'
            }
            onClick={() => toggleTeam('Warriors')}
          >
            Warriors
          </Button>
          <Button
            variant={selectedTeams.includes('Lakers') ? 'default' : 'secondary'}
            className={
              selectedTeams.includes('Lakers') ? 'bg-gray-600' : 'bg-gray-600'
            }
            onClick={() => toggleTeam('Lakers')}
          >
            Lakers
          </Button>
        </div>

        {/* Filters */}
        <div className="bg-gray-100 p-3 flex items-center space-x-2">
          <span className="font-medium">Filters:</span>
        </div>

        {/* 2D Projection */}
        <div className="bg-gray-50 p-4 h-80 relative">
          {clusters.map((cluster) => (
            <div key={cluster.id}>
              {/* Cluster ellipse */}
              <div
                className="absolute rounded-full border-2 border-dashed"
                style={{
                  left: `${cluster.plays[0].x - 70}px`,
                  top: `${cluster.plays[0].y - 50}px`,
                  width: '140px',
                  height: '100px',
                  backgroundColor: cluster.color,
                  borderColor: cluster.borderColor,
                }}
              />

              {/* Cluster label */}
              <div
                className="absolute text-center"
                style={{
                  left: `${cluster.plays[0].x - 10}px`,
                  top: `${cluster.plays[0].y}px`,
                  width: '80px',
                }}
              >
                {cluster.name}
              </div>

              {/* Play dots */}
              {cluster.plays.map((play) => (
                <div
                  key={play.id}
                  className={`absolute rounded-full cursor-pointer ${
                    selectedPlay === play.id ? 'ring-2 ring-black' : ''
                  }`}
                  style={{
                    left: `${play.x}px`,
                    top: `${play.y}px`,
                    width: '16px',
                    height: '16px',
                    backgroundColor: cluster.borderColor,
                  }}
                  onClick={() => setSelectedPlay(play.id)}
                />
              ))}
            </div>
          ))}
        </div>

        {/* Bottom panels */}
        <div className="grid grid-cols-2 gap-2">
          {/* Selected Play Panel */}
          <Card className="bg-gray-100 rounded-none">
            <CardHeader className="py-2 px-4">
              <CardTitle className="text-base">Selected Play</CardTitle>
            </CardHeader>
            <CardContent className="p-2">
              {/* Basketball court */}
              <div className="bg-amber-200 border border-gray-500 h-48 relative">
                {/* Court markings */}
                <div className="absolute left-1/2 top-1/2 w-16 h-16 border border-gray-500 rounded-full transform -translate-x-1/2 -translate-y-1/2" />
                <div className="absolute left-1/2 top-0 bottom-0 border-l border-gray-500 transform -translate-x-1/2" />
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-8 h-4 border border-gray-500" />
                <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-8 h-4 border border-gray-500" />

                {/* Players */}
                {courtPlayers.map((player) => (
                  <div
                    key={player.id}
                    className={`absolute rounded-full ${
                      player.team === 'offense' ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{
                      left: `${player.x}px`,
                      top: `${player.y}px`,
                      width: '10px',
                      height: '10px',
                    }}
                  />
                ))}
              </div>

              {/* Play actions */}
              <div className="mt-2 grid grid-cols-2 gap-2">
                <Button className="bg-blue-600 hover:bg-blue-700">Edit</Button>
                <Button className="bg-green-500 hover:bg-green-600">Tag</Button>
              </div>
            </CardContent>
          </Card>

          {/* Controls Panel */}
          <Card className="bg-gray-100 rounded-none">
            <CardHeader className="py-2 px-4">
              <CardTitle className="text-base">Clustering Controls</CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-4">
              {/* Similarity slider */}
              <div className="space-y-2">
                <div className="font-medium">Similarity Threshold</div>
                <Slider
                  value={similarityThreshold}
                  onValueChange={setSimilarityThreshold}
                  max={100}
                  step={1}
                />
              </div>

              {/* Cluster method dropdown */}
              <div className="space-y-2">
                <div className="font-medium">Cluster Method</div>
                <Select value={clusterMethod} onValueChange={setClusterMethod}>
                  <SelectTrigger className="bg-white">
                    <SelectValue placeholder="Select method" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="t-SNE">t-SNE</SelectItem>
                    <SelectItem value="UMAP">UMAP</SelectItem>
                    <SelectItem value="PCA">PCA</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Feature importance dropdown */}
              <div className="space-y-2">
                <div className="font-medium">Feature Importance</div>
                <Select
                  value={featureImportance}
                  onValueChange={setFeatureImportance}
                >
                  <SelectTrigger className="bg-white">
                    <SelectValue placeholder="Select feature" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Player Positions">
                      Player Positions
                    </SelectItem>
                    <SelectItem value="Ball Movement">Ball Movement</SelectItem>
                    <SelectItem value="Play Outcome">Play Outcome</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </div>
  );
}

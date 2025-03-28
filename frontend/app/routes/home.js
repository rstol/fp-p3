'use strict';
var __spreadArray =
  (this && this.__spreadArray) ||
  function (to, from, pack) {
    if (pack || arguments.length === 2)
      for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
          if (!ar) ar = Array.prototype.slice.call(from, 0, i);
          ar[i] = from[i];
        }
      }
    return to.concat(ar || Array.prototype.slice.call(from));
  };
Object.defineProperty(exports, '__esModule', { value: true });
exports.meta = meta;
exports.default = Home;
var card_1 = require('~/components/ui/card');
var button_1 = require('~/components/ui/button');
var react_1 = require('react');
var select_1 = require('~/components/ui/select');
var slider_1 = require('~/components/ui/slider');
function meta(_a) {
  return [
    { title: 'New React Router App' },
    { name: 'description', content: 'Welcome to React Router!' },
  ];
}
var clusters = [
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
var courtPlayers = [
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
function Home() {
  // State
  var _a = (0, react_1.useState)(null),
    selectedPlay = _a[0],
    setSelectedPlay = _a[1];
  var _b = (0, react_1.useState)([50]),
    similarityThreshold = _b[0],
    setSimilarityThreshold = _b[1];
  var _c = (0, react_1.useState)('t-SNE'),
    clusterMethod = _c[0],
    setClusterMethod = _c[1];
  var _d = (0, react_1.useState)('Player Positions'),
    featureImportance = _d[0],
    setFeatureImportance = _d[1];
  var _e = (0, react_1.useState)(['Warriors']),
    selectedTeams = _e[0],
    setSelectedTeams = _e[1];
  // Toggle team selection
  var toggleTeam = function (team) {
    if (selectedTeams.includes(team)) {
      setSelectedTeams(
        selectedTeams.filter(function (t) {
          return t !== team;
        }),
      );
    } else {
      setSelectedTeams(__spreadArray(__spreadArray([], selectedTeams, true), [team], false));
    }
  };
  return (
    <div className="mx-auto max-w-7xl py-6">
      <card_1.CardHeader className="pb-2">
        <card_1.CardTitle className="pb-2 text-center text-xl">
          Basketball Play Tactics Tool
        </card_1.CardTitle>
      </card_1.CardHeader>
      <card_1.CardContent className="space-y-4">
        {/* Teams Navigation */}
        <div className="flex items-center space-x-2 bg-blue-900 p-3">
          <span className="font-medium text-white">Teams</span>
          <button_1.Button
            variant={selectedTeams.includes('Warriors') ? 'default' : 'secondary'}
            className={selectedTeams.includes('Warriors') ? 'bg-blue-600' : 'bg-gray-600'}
            onClick={function () {
              return toggleTeam('Warriors');
            }}
          >
            Warriors
          </button_1.Button>
          <button_1.Button
            variant={selectedTeams.includes('Lakers') ? 'default' : 'secondary'}
            className={selectedTeams.includes('Lakers') ? 'bg-gray-600' : 'bg-gray-600'}
            onClick={function () {
              return toggleTeam('Lakers');
            }}
          >
            Lakers
          </button_1.Button>
        </div>

        {/* Filters */}
        <div className="flex items-center space-x-2 bg-gray-100 p-3">
          <span className="font-medium">Filters:</span>
        </div>

        {/* 2D Projection */}
        <div className="relative h-80 bg-gray-50 p-4">
          {clusters.map(function (cluster) {
            return (
              <div key={cluster.id}>
                {/* Cluster ellipse */}
                <div
                  className="absolute rounded-full border-2 border-dashed"
                  style={{
                    left: ''.concat(cluster.plays[0].x - 70, 'px'),
                    top: ''.concat(cluster.plays[0].y - 50, 'px'),
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
                    left: ''.concat(cluster.plays[0].x - 10, 'px'),
                    top: ''.concat(cluster.plays[0].y, 'px'),
                    width: '80px',
                  }}
                >
                  {cluster.name}
                </div>

                {/* Play dots */}
                {cluster.plays.map(function (play) {
                  return (
                    <div
                      key={play.id}
                      className={'absolute cursor-pointer rounded-full'.concat(
                        selectedPlay === play.id ? 'ring-2 ring-black' : '',
                      )}
                      style={{
                        left: ''.concat(play.x, 'px'),
                        top: ''.concat(play.y, 'px'),
                        width: '16px',
                        height: '16px',
                        backgroundColor: cluster.borderColor,
                      }}
                      onClick={function () {
                        return setSelectedPlay(play.id);
                      }}
                    />
                  );
                })}
              </div>
            );
          })}
        </div>

        {/* Bottom panels */}
        <div className="grid grid-cols-2 gap-2">
          {/* Selected Play Panel */}
          <card_1.Card className="rounded-none bg-gray-100">
            <card_1.CardHeader className="px-4 py-2">
              <card_1.CardTitle className="text-base">Selected Play</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="p-2">
              {/* Basketball court */}
              <div className="relative h-48 border border-gray-500 bg-amber-200">
                {/* Court markings */}
                <div className="absolute top-1/2 left-1/2 h-16 w-16 -translate-x-1/2 -translate-y-1/2 transform rounded-full border border-gray-500" />
                <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 transform border-l border-gray-500" />
                <div className="absolute top-0 left-1/2 h-4 w-8 -translate-x-1/2 transform border border-gray-500" />
                <div className="absolute bottom-0 left-1/2 h-4 w-8 -translate-x-1/2 transform border border-gray-500" />

                {/* Players */}
                {courtPlayers.map(function (player) {
                  return (
                    <div
                      key={player.id}
                      className={'absolute rounded-full'.concat(
                        player.team === 'offense' ? 'bg-red-500' : 'bg-blue-500',
                      )}
                      style={{
                        left: ''.concat(player.x, 'px'),
                        top: ''.concat(player.y, 'px'),
                        width: '10px',
                        height: '10px',
                      }}
                    />
                  );
                })}
              </div>

              {/* Play actions */}
              <div className="mt-2 grid grid-cols-2 gap-2">
                <button_1.Button className="bg-blue-600 hover:bg-blue-700">Edit</button_1.Button>
                <button_1.Button className="bg-green-500 hover:bg-green-600">Tag</button_1.Button>
              </div>
            </card_1.CardContent>
          </card_1.Card>

          {/* Controls Panel */}
          <card_1.Card className="rounded-none bg-gray-100">
            <card_1.CardHeader className="px-4 py-2">
              <card_1.CardTitle className="text-base">Clustering Controls</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-4 p-4">
              {/* Similarity slider */}
              <div className="space-y-2">
                <div className="font-medium">Similarity Threshold</div>
                <slider_1.Slider
                  value={similarityThreshold}
                  onValueChange={setSimilarityThreshold}
                  max={100}
                  step={1}
                />
              </div>

              {/* Cluster method dropdown */}
              <div className="space-y-2">
                <div className="font-medium">Cluster Method</div>
                <select_1.Select value={clusterMethod} onValueChange={setClusterMethod}>
                  <select_1.SelectTrigger className="bg-white">
                    <select_1.SelectValue placeholder="Select method" />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    <select_1.SelectItem value="t-SNE">t-SNE</select_1.SelectItem>
                    <select_1.SelectItem value="UMAP">UMAP</select_1.SelectItem>
                    <select_1.SelectItem value="PCA">PCA</select_1.SelectItem>
                  </select_1.SelectContent>
                </select_1.Select>
              </div>

              {/* Feature importance dropdown */}
              <div className="space-y-2">
                <div className="font-medium">Feature Importance</div>
                <select_1.Select value={featureImportance} onValueChange={setFeatureImportance}>
                  <select_1.SelectTrigger className="bg-white">
                    <select_1.SelectValue placeholder="Select feature" />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    <select_1.SelectItem value="Player Positions">
                      Player Positions
                    </select_1.SelectItem>
                    <select_1.SelectItem value="Ball Movement">Ball Movement</select_1.SelectItem>
                    <select_1.SelectItem value="Play Outcome">Play Outcome</select_1.SelectItem>
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
            </card_1.CardContent>
          </card_1.Card>
        </div>
      </card_1.CardContent>
    </div>
  );
}

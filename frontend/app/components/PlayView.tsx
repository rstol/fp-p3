import { Check, Edit } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '~/lib/stateStore';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { useFetcher } from 'react-router';
import { BASE_URL } from '~/lib/const';

interface ClientActionResult {
  success: boolean;
  error?: string;
  message?: string;
  data?: any; // Consider a more specific type for 'data' if known
}

function EditableField({
  id,
  label,
  value: initialValue,
  placeholder,
  isTextarea = false,
  onSave,
}: {
  id: string;
  label: string;
  value?: string | number;
  placeholder: string;
  isTextarea?: boolean;
  onSave: (newValue: string) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [currentValue, setCurrentValue] = useState(
    initialValue !== undefined ? String(initialValue) : '',
  );

  const inputRef = useRef<null | (HTMLTextAreaElement & HTMLInputElement)>(null);

  useEffect(() => {
    setCurrentValue(initialValue !== undefined ? String(initialValue) : '');
  }, [initialValue]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  function handleSaveInternal() {
    setIsEditing(false);
    onSave(currentValue);
  }

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    handleSaveInternal();
  }

  return (
    <div className="grid w-full items-center gap-1">
      <Label htmlFor={id} className="text-sm">
        {label}
      </Label>
      {!isEditing ? (
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-between border bg-gray-50 px-3! py-1! text-gray-700"
          onClick={() => setIsEditing(true)}
        >
          {currentValue ? currentValue : <span className="text-gray-400">{placeholder}</span>}
          <Edit size={6} />
        </Button>
      ) : (
        <form className="flex gap-2" onSubmit={handleSubmit}>
          {isTextarea ? (
            <Textarea
              id={id}
              placeholder={placeholder}
              value={currentValue}
              onChange={(e) => setCurrentValue(e.target.value)}
              className="flex-1"
              rows={1}
              ref={inputRef}
              onBlur={handleSaveInternal}
            />
          ) : (
            <Input
              id={id}
              type="text"
              placeholder={placeholder}
              value={currentValue}
              onChange={(e) => setCurrentValue(e.target.value)}
              className="flex-1"
              ref={inputRef}
              onBlur={handleSaveInternal}
            />
          )}
          <Button type="submit" size="sm" className="h-9">
            <Check size={6} />
          </Button>
        </form>
      )}
    </div>
  );
}

type PlayDetails = {
  videoURL: string;
};

export default function PlayView() {
  const selectedPlay = useDashboardStore((state) => state.selectedPlay);
  const stageSelectedPlayClusterUpdate = useDashboardStore(
    (state) => state.stageSelectedPlayClusterUpdate,
  );
  const stagedChangesCount = useDashboardStore((state) => state.stagedChangesCount);
  const pendingClusterUpdates = useDashboardStore((state) => state.pendingClusterUpdates);
  const selectedTeamId = useDashboardStore((state) => state.selectedTeamId);
  const clearPendingClusterUpdates = useDashboardStore((state) => state.clearPendingClusterUpdates);
  const [playDetails, setPlayDetails] = useState<PlayDetails>(null);

  const fetcher = useFetcher<ClientActionResult>();

  useEffect(() => {
    if (fetcher.state === 'idle' && fetcher.data) {
      const result = fetcher.data;
      if (result.success) {
        clearPendingClusterUpdates();
      } else if (result.error) {
        console.error('[PlayView.tsx] Fetcher submission error:', result.error);
      }
    }
  }, [fetcher.state, fetcher.data, clearPendingClusterUpdates]);

  useEffect(() => {
    if (!selectedPlay) return;

    const fetchPlayDetails = async () => {
      try {
        const res = await fetch(
          `${BASE_URL}/plays/${selectedPlay.game_id}/${selectedPlay.event_id}/video`,
        );
        const arrayBuffer = await res.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: 'video/mp4' });
        const videoURL = URL.createObjectURL(blob);
        setPlayDetails({ videoURL });
      } catch (error) {
        console.error('Failed to fetch play details:', error);
      }
    };

    fetchPlayDetails();
  }, [selectedPlay]);

  if (!selectedPlay) {
    return (
      <Card className="gap-4 border-none pt-1 shadow-none">
        <CardHeader>
          <CardTitle>Selected Play Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm">No play selected.</div>
        </CardContent>
      </Card>
    );
  }

  const onClickApplyChanges = async () => {
    if (!selectedTeamId) {
      console.error('[PlayView.tsx] No team selected. Cannot apply changes.');
      return;
    }
    if (pendingClusterUpdates.size === 0) {
      return;
    }

    const updatesToSend: { gameid: string; playid: number; cluster: number }[] = [];
    pendingClusterUpdates.forEach((newCluster, playIdKey) => {
      const parts = playIdKey.split('-');

      if (parts.length === 2) {
        const gameId = parts[0];
        const playIdStr = parts[1];
        const playIdNum = parseInt(playIdStr, 10);

        if (isNaN(playIdNum)) {
          console.error(
            `[PlayView.tsx] Failed to parse playId '${playIdStr}' to number for key '${playIdKey}'`,
          );
        } else {
          updatesToSend.push({
            gameid: gameId,
            playid: playIdNum,
            cluster: newCluster,
          });
        }
      } else {
        console.error(
          `[PlayView.tsx] Invalid playIdKey format: '${playIdKey}'. Expected 'GAMEID-PLAYID'.`,
        );
      }
    });

    if (updatesToSend.length === 0) {
      if (pendingClusterUpdates.size > 0) {
        console.error(
          '[PlayView.tsx] Changes were staged, but none could be parsed into valid updates. Aborting POST.',
        );
      }
      return;
    }

    const payload = {
      teamId: selectedTeamId,
      updates: updatesToSend,
    };

    fetcher.submit(payload, {
      method: 'POST',
      encType: 'application/json',
    });
  };

  const handleClusterChange = (newClusterValue: string) => {
    const newClusterId = parseInt(newClusterValue, 10);
    if (!isNaN(newClusterId)) {
      stageSelectedPlayClusterUpdate(newClusterId);
    } else {
      console.warn('Invalid Play Cluster ID entered, not a number:', newClusterValue);
    }
  };

  return (
    <Card className="gap-4 border-none pt-1 shadow-none">
      <CardHeader>
        <CardTitle>Selected Play Details</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {playDetails?.videoURL && (
          <video controls autoPlay disablePictureInPicture disableRemotePlayback loop muted>
            <source src={playDetails.videoURL} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        )}
        <div className="divide-y divide-solid text-sm">
          {selectedPlay.game_date && (
            <div className="flex gap-4 py-1">
              <span className="shrink-0">Game Date:</span>
              <span className="flex-1 text-right">
                {selectedPlay.game_date ? selectedPlay.game_date : 'N/A'}
              </span>
            </div>
          )}
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Description Home:</span>
            <span className="flex-1 text-right">{selectedPlay.event_desc_home}</span>
          </div>
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Description Away:</span>
            <span className="flex-1 text-right">{selectedPlay.event_desc_away}</span>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2">
        <EditableField
          id="play_cluster"
          label="Play Cluster"
          placeholder="Set Cluster ID..."
          value={selectedPlay.cluster}
          onSave={handleClusterChange}
        />
        <EditableField
          id="play_note"
          label="Play Note"
          placeholder="Add a note..."
          isTextarea={true}
          onSave={(newNote) => {
            console.log('Play Note saved:', newNote);
            // TODO: Implement saving logic for play note (update store, API call)
            // Example: if (selectedPlay) { updateSelectedPlayNote(newNote); }
          }}
        />

        {stagedChangesCount > 0 && (
          <div className="mt-6 flex w-full items-center justify-end border-t pt-4">
            <Button
              size="sm"
              disabled={stagedChangesCount === 0 || !selectedTeamId}
              onClick={onClickApplyChanges}
            >
              Apply Changes ({stagedChangesCount})
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
}

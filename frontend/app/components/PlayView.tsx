import { Check, Edit, Tag, Save } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '~/lib/stateStore';
import { PlayDetailsSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import type { DetailedPlay } from '~/types/data';
import { Badge } from '~/components/ui/badge';
import { getTag, setTag } from '~/lib/tag-store';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '~/components/ui/select';

const AVAILABLE_TAGS: string[] = ['0', '1', '2', '3', '4', 'None'];

function EditableField({
  id,
  label,
  placeholder,
  isTextarea = false,
  initialValue = '',
  onSave,
}: {
  id: string;
  label: string;
  placeholder: string;
  isTextarea?: boolean;
  initialValue?: string;
  onSave?: (value: string) => Promise<void> | void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(initialValue);
  const [isSaving, setIsSaving] = useState(false);

  const inputRef = useRef<null | (HTMLTextAreaElement & HTMLInputElement)>(null);

  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  async function handleSave() {
    if (onSave && value !== initialValue) {
      setIsSaving(true);
      try {
        await onSave(value);
      } catch (error) {
        console.error('Error saving value:', error);
      } finally {
        setIsSaving(false);
      }
    }
    setIsEditing(false);
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
          disabled={isSaving}
        >
          {value ? value : <span className="text-gray-400">{placeholder}</span>}
          <Edit size={15} />
        </Button>
      ) : (
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); handleSave(); }} onBlur={handleSave}>
          {isTextarea ? (
            <Textarea
              id={id}
              placeholder={placeholder}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="flex-1"
              rows={1}
              ref={inputRef}
              disabled={isSaving}
            />
          ) : (
            <Input
              id={id}
              type="text"
              placeholder={placeholder}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="flex-1"
              ref={inputRef}
              disabled={isSaving}
            />
          )}
          <Button onClick={handleSave} size="sm" className="h-9" disabled={isSaving}>
            {isSaving ? <span className="animate-pulse">...</span> : <Check size={15} />}
          </Button>
        </form>
      )}
    </div>
  );
}

export default function PlayView() {
  const selectedPlay = useDashboardStore((state) => state.selectedPlay);
  const pendingEdits = useDashboardStore((state) => state.pendingEdits);
  const updatePlay = useDashboardStore((state) => state.updatePlay);
  const setScatterPoints = useDashboardStore((state) => state.setScatterPoints);
  const updatePendingEditTag = useDashboardStore((state) => state.updatePendingEditTag);
  const clearPendingEdits = useDashboardStore((state) => state.clearPendingEdits);
  const addPendingEdit = useDashboardStore((state) => state.addPendingEdit);
  const [playDetails, setPlayDetails] = useState<DetailedPlay | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tag, setTagState] = useState<string>('');
  const [originalCluster, setOriginalCluster] = useState<string>('');
  const [isApplying, setIsApplying] = useState(false);
  const [changeApplied, setChangeApplied] = useState(false);

  // Fetch detailed play data when selectedPlay changes
  useEffect(() => {
    if (!selectedPlay) {
      setPlayDetails(null);
      setTagState('');
      setOriginalCluster('');
      setChangeApplied(false);
      return;
    }

    // Store the original cluster value when a point is selected
    setOriginalCluster(`${selectedPlay.cluster}`);
    // Reset the change applied state when a new point is selected
    setChangeApplied(false);

    async function fetchPlayDetails() {
      setIsLoading(true);
      setError(null);
      try {
        if (!selectedPlay) {
          throw new Error("No play selected");
        }

        const gameId = selectedPlay.game_id;
        const eventId = selectedPlay.event_id;

        const response = await fetch(
          `/api/v1/games/${gameId}/plays/${eventId}`
        );

        if (!response.ok) {
          throw new Error(`Error fetching play details: ${response.statusText}`);
        }

        const data = await response.json();
        setPlayDetails(data);

        // Get tag from local storage instead of backend
        const storedTag = getTag(gameId, eventId) || String(selectedPlay.cluster);
        setTagState(storedTag);
      } catch (error) {
        console.error('Error fetching play details:', error);
        setError('Failed to load play details');
      } finally {
        setIsLoading(false);
      }
    }

    fetchPlayDetails();
  }, [selectedPlay]);

  async function handleSaveTag(value: string) {
    if (!selectedPlay) return;

    try {
      // Standardize the value - numbers for cluster tags
      let standardizedValue = value.trim();
      try {
        // If it's a number, keep it as string but make sure it's valid
        parseInt(standardizedValue);
      } catch (e) {
        // If not a number, use as-is (custom tag)
        standardizedValue = value.trim();
      }

      const gameId = selectedPlay.game_id;
      const eventId = selectedPlay.event_id;

      // Save tag to local storage instead of sending to backend
      setTag(gameId, eventId, standardizedValue);

      // Update state with new tag
      setTagState(standardizedValue);

      // Reset change applied flag since we made a new change
      setChangeApplied(false);

      // Notify other components that tags have changed
    } catch (error) {
      console.error('Error saving tag:', error);
    }
  }

  const handleTagChange = (newTag: string) => {
    if (selectedPlay) {
      // If the point isn't in pending edits yet, add it first
      if (!pendingEdits.find(edit => edit.point_id === String(selectedPlay.event_id))) {
        const newEdit = {
          point_id: String(selectedPlay.event_id),
          game_id: String(selectedPlay.game_id),
          new_cluster: newTag,
          original_cluster: selectedPlay.cluster ? String(selectedPlay.cluster) : 'None',
        };
        // Directly add using the function from the hook (assuming addPendingEdit exists)
        addPendingEdit(newEdit);
      } else {
        // Otherwise, just update the tag for the existing pending edit
        updatePendingEditTag(String(selectedPlay.event_id), newTag);
      }
    }
  };

  const handleApplyPendingChanges = async () => {
    if (pendingEdits.length === 0 || isApplying) return;
    setIsApplying(true);

    console.log("Applying pending changes:", pendingEdits);

    try {
      const gameId = pendingEdits[0]?.game_id; // Assuming all edits are for the same game
      if (!gameId) {
        console.error('Cannot apply changes: gameId is missing.');
        return;
      }

      // Prepare data for the backend
      const updates = pendingEdits.map((edit) => ({
        point_id: edit.point_id,
        game_id: edit.game_id,
        new_cluster: edit.new_cluster,
        original_cluster: edit.original_cluster,
      }));

      const response = await fetch(`/api/v1/update-clustering`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ game_id: gameId, updates: updates }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
      }

      // IMPORTANT: Get the updated points from the response
      const result = await response.json();

      if (result.success && result.points) {
        // Update the state store with the new points
        setScatterPoints(result.points);
        // Clear pending edits
        clearPendingEdits();
        // Optionally: Add user feedback (e.g., toast notification)
        console.log('Cluster changes applied successfully.');
      } else {
        throw new Error('Backend indicated success=false or points were missing.');
      }
    } catch (error) {
      console.error('Error applying pending changes:', error);
      // TODO: Add user feedback (e.g., toast notification)
    } finally {
      setIsApplying(false);
    }
  };

  if (isLoading) {
    return <PlayDetailsSkeleton />;
  } else if (error) {
    return (
      <Card className="gap-4 border-none pt-1 shadow-none">
        <CardHeader>
          <CardTitle>Selected Play Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-sm text-red-500">{error}</div>
        </CardContent>
      </Card>
    );
  } else if (!selectedPlay) {
    return (
      <Card className="gap-4 border-none pt-1 shadow-none">
        <CardHeader>
          <CardTitle>Selected Play Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-sm">No play selected.</div>
        </CardContent>
      </Card>
    );
  }

  const play = selectedPlay!;

  const currentPendingEdit = pendingEdits.find(edit => edit.point_id === String(selectedPlay.event_id));
  const displayCluster = currentPendingEdit
    ? currentPendingEdit.new_cluster
    : selectedPlay.cluster ? String(selectedPlay.cluster) : 'None';

  return (
    <Card className="gap-4 border-none pt-1 shadow-none">
      <CardHeader>
        <CardTitle>Selected Play Details</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <img src="/spurs.gif" alt="play_visualization" />
        <div className="divide-y divide-solid text-sm">
          {/* Game information */}
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Game ID:</span>
            <span className="flex-1 text-right">{playDetails?.game_id || play.game_id}</span>
          </div>
          <div className="flex gap-4 pb-1">
            <span className="shrink-0">Period:</span>
            <span className="flex-1 text-right">{playDetails?.period || 'N/A'}</span>
          </div>

          {/* Play descriptions */}
          <div className="flex gap-4 py-1">
            <span className="shrink-0">Description Home:</span>
            <span className="flex-1 text-right">
              {playDetails?.event_desc_home || 'N/A'}
            </span>
          </div>
          <div className="flex gap-4 pt-1">
            <span className="shrink-0">Description Away:</span>
            <span className="flex-1 text-right">
              {playDetails?.event_desc_away || 'N/A'}
            </span>
          </div>

          {/* Play type */}
          <div className="flex gap-4 pt-1">
            <span className="shrink-0">Play Type:</span>
            <span className="flex-1 text-right">
              {play.play_type || 'Unknown'}
            </span>
          </div>

          {/* Tags display */}
          <div className="flex flex-wrap gap-2 pt-2">
            <span className="shrink-0 flex items-center">
              <Tag size={14} className="mr-1" /> Tag:
            </span>
            <div className="flex flex-wrap gap-1 flex-1 justify-end">
              {tag ? (
                <Badge
                  key={tag}
                  className="bg-blue-100 text-blue-800"
                >
                  {tag}
                </Badge>
              ) : (
                <span className="text-gray-400">No tag assigned</span>
              )}
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex flex-col gap-4">
        <div className="mt-4">
          <label htmlFor="tag-select" className="block text-sm font-medium text-gray-700 mb-1">Cluster Tag:</label>
          <Select
            value={displayCluster}
            onValueChange={handleTagChange}
          >
            <SelectTrigger id="tag-select">
              <SelectValue placeholder="Select tag" />
            </SelectTrigger>
            <SelectContent>
              {AVAILABLE_TAGS.map((tag: string) => (
                <SelectItem key={tag} value={tag}>
                  {tag}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <EditableField
          id="play_note"
          label="Play Note"
          placeholder="Add a note..."
          isTextarea={true}
        />
        <Button
          onClick={handleApplyPendingChanges}
          disabled={pendingEdits.length === 0 || isApplying}
          className="w-full"
        >
          {isApplying ? 'Applying...' : `Apply ${pendingEdits.length} Pending Change(s)`}
        </Button>
      </CardFooter>
    </Card>
  );
}

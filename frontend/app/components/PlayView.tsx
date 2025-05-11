import { Check, Edit } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '~/lib/stateStore';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';

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

export default function PlayView() {
  const selectedPlay = useDashboardStore((state) => state.selectedPlay);
  const updateSelectedPlayCluster = useDashboardStore((state) => state.updateSelectedPlayCluster);

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

  return (
    <Card className="gap-4 border-none pt-1 shadow-none">
      <CardHeader>
        <CardTitle>Selected Play Details</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <img src="/spurs.gif" alt="spurs_play_view" />
        <div className="divide-y divide-solid text-sm">
          {selectedPlay.game_date && (
            <div className="flex gap-4 py-1">
              <span className="shrink-0">Game Date:</span>
              <span className="flex-1 text-right">{selectedPlay.game_date ? selectedPlay.game_date : 'N/A'}</span>
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
          onSave={(newClusterValue) => {
            const newClusterId = parseInt(newClusterValue, 10);
            if (!isNaN(newClusterId)) {
              updateSelectedPlayCluster(newClusterId);
              console.log('Play Cluster saved and store updated:', newClusterId);
              // TODO: Implement API call to persist this change to the backend
            } else {
              console.warn('Invalid Play Cluster ID entered, not a number:', newClusterValue);
            }
          }}
        />
        <EditableField
          id="play_note"
          label="Play Note"
          placeholder="Add a note..."
          isTextarea={true}
          // value={selectedPlay.note || ''} 
          onSave={(newNote) => {
            console.log('Play Note saved:', newNote);
            // TODO: Implement saving logic for play note (update store, API call)
            // Example: if (selectedPlay) { updateSelectedPlayNote(newNote); }
          }}
        />
      </CardFooter>
    </Card>
  );
}

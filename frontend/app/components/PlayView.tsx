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
  placeholder,
  isTextarea = false,
}: {
  id: string;
  label: string;
  placeholder: string;
  isTextarea?: boolean;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState('');

  const inputRef = useRef<null | (HTMLTextAreaElement & HTMLInputElement)>(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  function handleSave() {
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
        >
          {value ? value : <span className="text-gray-400">{placeholder}</span>}
          <Edit size={6} />
        </Button>
      ) : (
        <form className="flex gap-2" onSubmit={handleSave} onBlur={handleSave}>
          {isTextarea ? (
            <Textarea
              id={id}
              placeholder={placeholder}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="flex-1"
              rows={1}
              ref={inputRef}
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
            />
          )}
          <Button onClick={handleSave} size="sm" className="h-9">
            <Check size={6} />
          </Button>
        </form>
      )}
    </div>
  );
}

export default function PlayView() {
  const selectedPlay = useDashboardStore((state) => state.selectedPlay);

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
        <EditableField id="play_tag" label="Play Tag" placeholder="Tag the play..." />
        <EditableField
          id="play_note"
          label="Play Note"
          placeholder="Add a note..."
          isTextarea={true}
        />
      </CardFooter>
    </Card>
  );
}

import { Check, Edit } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from '~/lib/stateStore';
import { PlayDetailsSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import type { DetailedPlay } from '~/types/data';

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
  const [playDetails, setPlayDetails] = useState<DetailedPlay | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch detailed play data when selectedPlay changes
  useEffect(() => {
    if (!selectedPlay) {
      setPlayDetails(null);
      return;
    }

    async function fetchPlayDetails() {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `/api/v1/games/${selectedPlay.game_id}/plays/${selectedPlay.event_id}`
        );
        
        if (!response.ok) {
          throw new Error(`Error fetching play details: ${response.statusText}`);
        }
        
        const data = await response.json();
        setPlayDetails(data);
      } catch (error) {
        console.error('Error fetching play details:', error);
        setError('Failed to load play details');
      } finally {
        setIsLoading(false);
      }
    }

    fetchPlayDetails();
  }, [selectedPlay]);

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
            <span className="flex-1 text-right">{playDetails?.period || play.period || 'N/A'}</span>
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
          
          {/* Play type and cluster */}
          <div className="flex gap-4 pt-1">
            <span className="shrink-0">Play Type:</span>
            <span className="flex-1 text-right">
              {play.play_type || 'Unknown'}
            </span>
          </div>
          <div className="flex gap-4 pt-1">
            <span className="shrink-0">Cluster:</span>
            <span className="flex-1 text-right">
              {play.cluster !== undefined ? play.cluster : 'N/A'}
            </span>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex flex-col gap-4">
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

/** Center column: one long-lived <video> over the match's source. Selecting a
 *  clip seeks it; nothing remounts, so the source is fetched once. The clip's
 *  end only advances when continuous play is on — scrubbing past it is fine. */

import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import type { Clip } from './types';
import { KIND_LABEL, fmtTime, playerLabel } from './types';

function seekVideo(
  v: HTMLVideoElement | null,
  t: number,
  play: boolean,
  onError: (message: string) => void,
) {
  if (!v || v.readyState < HTMLMediaElement.HAVE_METADATA) return;
  v.currentTime = t;
  if (play) void v.play().catch(() => onError('無法自動播放，請按播放鍵。'));
}

export function ClipPlayer({
  src,
  clip,
  onEnded,
}: {
  src: string;
  clip: Clip | null;
  /** Called when a clip finishes under continuous play. */
  onEnded: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [now, setNow] = useState(0);
  const [continuous, setContinuous] = useState(false);
  const [error, setError] = useState('');
  // Set on each clip change so the end of the *new* clip is what advances,
  // not a stale timeupdate from before the seek landed.
  const armed = useRef(false);

  const seek = (t: number, play: boolean) => seekVideo(video.current, t, play, setError);

  // react-query's structural sharing keeps `clip` identical across refetches
  // that change nothing, so this fires on a real selection or reframing only.
  useEffect(() => {
    if (!clip) return;
    armed.current = true;
    seekVideo(video.current, clip.start, true, setError);
  }, [clip]);

  const span = clip ? Math.max(clip.end - clip.start, 0.001) : 1;
  const pct = (t: number) =>
    `${Math.min(100, Math.max(0, ((t - (clip?.start ?? 0)) / span) * 100))}%`;

  return (
    <div className="space-y-3">
      <video
        ref={video}
        className="aspect-video w-full rounded-xl bg-black"
        controls
        preload="metadata"
        src={src}
        onLoadedMetadata={() => {
          setError('');
          if (clip) seek(clip.start, false);
        }}
        onError={() => setError('影片讀取失敗：此比賽可能沒有來源影片或 R2 無法讀取。')}
        onTimeUpdate={(e) => {
          const t = e.currentTarget.currentTime;
          setNow(t);
          if (clip && continuous && armed.current && t >= clip.end) {
            armed.current = false;
            onEnded();
          }
        }}
      />
      {error && (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      )}
      {clip ? (
        <>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span className="font-semibold">
              {KIND_LABEL[clip.kind] ?? clip.kind}
              {clip.rally_index != null && ` · Rally ${clip.rally_index}`}
            </span>
            <span className="font-mono text-xs text-text-muted">
              {fmtTime(clip.start)} – {fmtTime(clip.end)}
            </span>
            <span className="ml-auto flex items-center gap-3">
              <label className="flex items-center gap-1.5 text-xs text-text-secondary">
                <input
                  type="checkbox"
                  checked={continuous}
                  onChange={(e) => setContinuous(e.target.checked)}
                />
                連續播放
              </label>
              <Button size="sm" onClick={() => seek(clip.start, true)}>
                重播片段
              </Button>
            </span>
          </div>
          {/* The clip's own timeline: playhead plus every touch it frames. */}
          <div className="relative h-8 rounded-md bg-surface-200">
            <div
              className="absolute inset-y-0 left-0 rounded-md bg-primary/15"
              style={{ width: pct(now) }}
            />
            {clip.touches.map((t, i) => (
              <button
                key={`${t.event_id}:${i}`}
                title={`${KIND_LABEL[t.kind] ?? t.kind} ${fmtTime(t.time)}${t.player ? ` ${playerLabel(t.player)}` : ''}`}
                className="absolute top-1 h-6 w-1.5 -translate-x-1/2 rounded-full bg-accent hover:scale-x-150"
                style={{ left: pct(t.time) }}
                onClick={() => seek(t.time, false)}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {clip.touches.map((t, i) => (
              <button
                key={`${t.event_id}:${i}`}
                onClick={() => seek(t.time, false)}
                className={cn(
                  'rounded-md border border-border-light px-2 py-1 text-xs',
                  Math.abs(now - t.time) < 0.25
                    ? 'border-accent text-accent'
                    : 'text-text-secondary',
                )}
              >
                {KIND_LABEL[t.kind] ?? t.kind} {fmtTime(t.time)}
                {t.player && ` · ${playerLabel(t.player)}`}
              </button>
            ))}
          </div>
        </>
      ) : (
        <p className="text-sm text-text-muted">從右側選一個片段開始播放。</p>
      )}
    </div>
  );
}

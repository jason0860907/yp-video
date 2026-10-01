/** The Label players' transport row and its footnotes: time / frame readout,
 *  Play and frame steps (a panel's own buttons follow), the fps · frames
 *  line, and the keyboard hint strip. */

import { Fragment, type ReactNode } from 'react';
import { Button } from '@/components/ui/Button';
import { formatActionTime } from '@/lib/actionEditorModel';

export function PlayerTransport({
  frame,
  fps,
  playing,
  onTogglePlay,
  onStep,
  children,
}: {
  frame: number;
  fps: number;
  playing: boolean;
  onTogglePlay: () => void;
  onStep: (frames: number) => void;
  /** The panel's own controls, after the frame steps. */
  children?: ReactNode;
}) {
  return (
    <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
      <span className="rounded-lg border border-border bg-surface-200/50 px-2.5 py-1 font-mono text-sm tabular-nums text-text-primary">
        {formatActionTime(frame / (fps || 30))} / f{frame}
      </span>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={onTogglePlay}>
          {playing ? 'Pause' : 'Play'}
        </Button>
        <Button size="sm" onClick={() => onStep(-1)}>
          ◂
        </Button>
        <Button size="sm" onClick={() => onStep(1)}>
          ▸
        </Button>
        {children}
      </div>
    </div>
  );
}

/** `29.970 fps · 54000 frames`; empty until both are known. */
export function FrameStats({ fps, numFrames }: { fps: number; numFrames: number }) {
  return (
    <span className="font-mono text-[11px] tabular-nums text-text-muted">
      {fps && numFrames ? `${fps.toFixed(3)} fps · ${numFrames} frames` : ''}
    </span>
  );
}

/** `Space play · ← → frame · …` under a player. */
export function KeyHints({ keys }: { keys: [key: string, does: string][] }) {
  return (
    <p className="px-1 text-[11px] text-text-muted">
      {keys.map(([key, does], i) => (
        <Fragment key={key}>
          {i > 0 && ' · '}
          <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">
            {key}
          </kbd>{' '}
          {does}
        </Fragment>
      ))}
    </p>
  );
}

/** Every actor position the server computed, by rally — the Action Label
 *  list with court x, y and the ball's height in place of the editors — plus
 *  a download of the same JSON. Clicking a row seeks the video there. */

import { useEffect, useRef, useState } from 'react';
import { API, apiUrl, errMsg } from '@/lib/api';
import { formatActionTime, OUTSIDE_RALLY_KEY } from '@/lib/actionEditorModel';
import { scrollActionIntoView, scrollRallyTop } from '@/lib/sidebarScroll';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { ActionDot, EventRows, OutsideRow, RallyRow } from '@/components/action/RallyRows';
import type { Rally } from '@/components/labeling/shared';
import type { CourtPositions } from './geometry';

type Position = CourtPositions['events'][number];

const metres = (v: number | undefined) => (v == null ? '–' : v.toFixed(2));

export function PositionsList({
  video,
  positions,
  rallies,
  error,
  time,
  currentId,
  onSeek,
}: {
  video: string;
  positions: Position[];
  rallies: Rally[];
  error: unknown;
  /** The playhead, seconds. */
  time: number;
  /** The touch drawn on the video right now. */
  currentId: string | null;
  onSeek: (seconds: number) => void;
}) {
  const list = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const liveRally = rallies.find((r) => time >= r.start && time < r.end)?.rally_id ?? null;

  // Entering a rally opens its group; a collapse mid-rally sticks.
  const [enteredRally, setEnteredRally] = useState<number | null>(null);
  if (liveRally !== enteredRally) {
    setEnteredRally(liveRally);
    if (liveRally != null) setExpanded(String(liveRally));
  }
  // Keep the touch on the video in view as it plays.
  useEffect(() => {
    if (currentId) scrollActionIntoView(list.current, currentId);
  }, [currentId]);

  const byRally = (id: number) => positions.filter((p) => p.rally_id === id);
  const outside = positions.filter((p) => p.rally_id == null);
  const openRally = (rally: Rally) => {
    setExpanded(String(rally.rally_id));
    scrollRallyTop(list.current, rally.rally_id);
    onSeek(rally.start);
  };

  const rows = (entries: Position[], empty: string) => (
    <EventRows
      entries={entries}
      empty={empty}
      columns="minmax(4rem,1fr) 2.6rem 2.6rem 2.6rem 2.6rem"
      selectedId={currentId}
      isActive={(p) => Math.abs(p.time - time) <= 0.5}
      onJump={(p) => onSeek(p.time)}
      rowClassName={(p) => !p.in_court && 'opacity-60'}
    >
      {(p) => (
        <>
          <span className="flex min-w-0 items-center gap-1.5">
            <ActionDot label={p.label} visible={p.ball_image != null} />
            <span className="truncate text-xs text-text-primary">{p.label ?? '—'}</span>
          </span>
          <span className="text-center font-heading text-[10px] tabular-nums text-text-muted">
            {formatActionTime(p.time)}
          </span>
          <span className="text-right font-mono text-[11px] tabular-nums" title="x (m) — along the court">
            {metres(p.court_xy[0])}
          </span>
          <span className="text-right font-mono text-[11px] tabular-nums" title="y (m) — across the court">
            {metres(p.court_xy[1])}
          </span>
          <span className="text-right font-mono text-[11px] tabular-nums" title="z (m) — ball height at the touch">
            {metres(p.ball_3d?.[2])}
          </span>
        </>
      )}
    </EventRows>
  );

  return (
    <>
      <div className="mb-2.5 flex items-center justify-between">
        <SectionLabel className="mb-0">
          Rallies ({rallies.length} rally · {positions.length} action)
        </SectionLabel>
        {positions.length > 0 && (
          <a
            href={apiUrl(API.court.positions(video))}
            download={`${video.replace(/\.[^.]+$/, '')}_court_positions.json`}
            className="text-[11px] text-primary hover:underline"
          >
            Download JSON
          </a>
        )}
      </div>
      {error ? (
        <p className="text-[11px] text-amber-400">{errMsg(error)}</p>
      ) : (
        <div ref={list} className="max-h-[32rem] space-y-1.5 overflow-y-auto pr-1">
          <p className="text-right font-mono text-[10px] text-text-muted/70">x · y · z (m)</p>
          {rallies.map((rally, ri) => {
            const entries = byRally(rally.rally_id);
            const isOpen = expanded === String(rally.rally_id);
            return (
              <div key={rally.rally_id} className="space-y-1.5">
                <RallyRow
                  index={ri}
                  rally={rally}
                  count={entries.length}
                  open={isOpen}
                  selected={isOpen}
                  live={rally.rally_id === liveRally}
                  onSelect={() => openRally(rally)}
                  onToggle={() => (isOpen ? setExpanded(null) : openRally(rally))}
                />
                {isOpen && rows(entries, 'No positions in this rally')}
              </div>
            );
          })}
          {outside.length > 0 && (
            <div className="space-y-1.5">
              <OutsideRow
                rowKey={OUTSIDE_RALLY_KEY}
                count={outside.length}
                open={expanded === OUTSIDE_RALLY_KEY}
                onToggle={() => setExpanded(expanded === OUTSIDE_RALLY_KEY ? null : OUTSIDE_RALLY_KEY)}
              />
              {expanded === OUTSIDE_RALLY_KEY && rows(outside, 'No outside positions')}
            </div>
          )}
        </div>
      )}
    </>
  );
}

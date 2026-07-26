/** Types and helpers shared by the ReID Label page, its video player and
 *  its group board. */

import { ApiError } from '@/lib/api';
import type { ReidRecord } from '@/types/api';

/** The human verdict on one event's actor. "unreviewed" is the absence of
 *  one — never inferred from the record's shape (see actor/labels.py). */
export type ActorVerdict = NonNullable<ReidRecord['actor_review']>;

export const verdictOf = (r: Pick<ReidRecord, 'actor_review'>): ActorVerdict =>
  r.actor_review ?? 'unreviewed';

/** One vocabulary for every surface that shows a verdict — the sidebar row,
 *  the picker panel and the review list. Three views inventing their own
 *  wording is how "confirmed" ends up meaning two different things. */
export const VERDICT: Record<ActorVerdict, { label: string; glyph: string; title: string }> = {
  unreviewed: { label: 'Unreviewed', glyph: '·', title: 'Nobody has ruled on this actor yet' },
  confirmed_auto: { label: 'Confirmed', glyph: '✓', title: 'Automatic pick, confirmed by a human' },
  manual: { label: 'Manual', glyph: '✎', title: 'Actor picked by hand' },
  occluded: { label: 'Occluded', glyph: '⊘', title: 'Nobody in frame is the actor' },
};

/** What the automatic policy thought, for an event nobody has ruled on yet.
 *
 *  A model's abstention is stored as `unresolved` — the same state as "the
 *  geometry found nobody" — because only a human may write the `occluded`
 *  VERDICT. That keeps a guess from ever looking like a conclusion, but it
 *  also means the model's reason is invisible unless something reads it back
 *  out of the diagnostic. This is that something.
 *
 *  It is a HINT: it tells the reviewer where to look, never what to record. */
export const HINT: Record<string, { label: string; title: string }> = {
  occluded: {
    label: 'Model: occluded?',
    title: 'The model saw no visible player performing this — confirm or override',
  },
  untracked: {
    label: 'Model: not tracked',
    title:
      'The model believes someone acted but tracking has no box for them here. ' +
      'Re-running Rally Tracking may fix this; relabelling will not.',
  },
};

export interface ActorHint {
  label: string;
  title: string;
  confidence?: number;
}

export const hintOf = (
  r: Pick<ReidRecord, 'actor_review' | 'association'>,
): ActorHint | null => {
  if (verdictOf(r) !== 'unreviewed') return null;
  const kind = r.association?.kind;
  if (!kind || kind === 'track') return null;
  const hint = HINT[kind];
  if (!hint) return null;
  return { ...hint, confidence: r.association?.confidence ?? undefined };
};

export interface Rally {
  rally_id: number;
  start: number;
  end: number;
}

/** One sidebar row: an action event's time, whether or not it has a ReID
 *  record (score / non-visible events have none — no box, just the time). */
export interface SidebarAction {
  id: string;
  frame: number;
  time: number | null;
  label?: string;
  visible: boolean;
}

/** Explicit actor-fix commands; invalid combinations are unrepresentable. */
export type ActorFix =
  | {
      mode: 'pick';
      box: [number, number, number, number];
      /** The tracklet clicked, "{rally_id}:{track_id}". When set the box is
       *  only an anchor — the server re-resolves the tracklet to a croppable
       *  detection, so the crop stays reproducible from the label alone. */
      track?: string;
      /** Cross-frame pick: crop pixels from this frame. Box picks only. */
      frame?: number;
      /** False when no stored detection is this player. Box picks only. */
      snap?: boolean;
    }
  | { mode: 'occluded' }
  | { mode: 'auto' };

/** ByteTrack tracklets + which tracklet each event's actor sits on. */
export interface TrackData {
  tracklets: { rally_id: number; track_id: number; frames: number[]; boxes: [number, number, number, number][] }[];
  links: Record<string, { rally_id: number; track_id: number }>;
}

/** GET /tracklets/masks — one rally's masks, whole tracklets at once.
 *  Values are base64 packed bits (box-crop space, ``mask_hw`` grid), row i
 *  aligned with the tracklet's i-th frame from /tracklets/{name}. */
export interface TrackMasks {
  mask_hw: [number, number];
  tracks: Record<string, string>;
}

/** Whether a human could endorse the policy's answer here.
 *
 *  Two answers are endorsable and they land as different verdicts: a PICK
 *  becomes `confirmed_auto`, and an explicit "nobody is visible" becomes
 *  `occluded`. Merely abstaining is neither — `untracked` says somebody acted
 *  and tracking lost them, which re-running tracking may fix. */
export const canConfirm = (
  r: Pick<ReidRecord, 'resolution' | 'actor_review' | 'association'>,
) =>
  verdictOf(r) === 'unreviewed' &&
  (r.resolution === 'auto' ||
    (r.resolution === 'unresolved' && r.association?.kind === 'occluded'));

/** The rally an event falls in, or null when it sits between rallies.
 *
 *  The one rule, shared by the player's sidebar and the Association review
 *  list: prefer the stored time, fall back to frame/fps. Two views disagreeing
 *  about which rally an event belongs to would be invisible until a confirm
 *  landed on the wrong group. */
export const rallyOf = <T extends { frame: number; time?: number | null }>(
  rallies: Rally[],
  event: T,
  fps: number,
): Rally | null => {
  const t = event.time != null ? event.time : event.frame / fps;
  return rallies.find((r) => t >= r.start && t <= r.end) ?? null;
};

/** The tracklet an event's actor sits on, as a stable "rally:track" key. */
export const trackKeyOf = (links: TrackData['links'], id: string) => {
  const l = links[id];
  return l ? `${l.rally_id}:${l.track_id}` : null;
};

/** Stable hue per tracklet, shared by the video overlay and crop badges. */
export const trackColor = (key: string) => {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} 75% 62%)`;
};

export const fmtTime = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

/** Types and helpers shared by the ReID Label page, its video player and
 *  its group board. */

import { rallyAt } from '@/lib/actionEditorModel';
import type { ReidRecord } from '@/types/api';

/** The verdict on one event's actor (actor/labels.py). "unreviewed" is the
 *  absence of one — never inferred from the record's shape. `agent` and
 *  `agent_occluded` are an annotation agent's proposals, not a human's. */
export type ActorVerdict = NonNullable<ReidRecord['actor_review']>;

export const verdictOf = (r: Pick<ReidRecord, 'actor_review'>): ActorVerdict =>
  r.actor_review ?? 'unreviewed';

/** One vocabulary for every surface that shows a verdict — the sidebar row,
 *  the picker panel and the review list. Three views inventing their own
 *  wording is how "confirmed" ends up meaning two different things. */
export const VERDICT: Record<ActorVerdict, { label: string; glyph: string; title: string }> = {
  unreviewed: { label: 'Unreviewed', glyph: '·', title: 'Nobody has ruled on this actor yet' },
  confirmed: { label: 'Confirmed', glyph: '✓', title: 'A proposed pick (automatic or agent), confirmed by a human' },
  manual: { label: 'Manual', glyph: '✎', title: 'Actor picked by hand' },
  occluded: { label: 'Occluded', glyph: '⊘', title: 'A human says nobody in frame is the actor' },
  agent: { label: 'Agent', glyph: '✦', title: "An annotation agent's pick — Confirm it, or pick the right person" },
  agent_occluded: { label: 'Agent: occluded', glyph: '✦', title: 'An annotation agent says nobody in frame is the actor — Confirm it, or pick the right person' },
};

/** Whether an event still needs a human: nobody ruled on it, or only an
 *  agent did. */
export const needsReview = (v: ActorVerdict) => v === 'unreviewed' || v === 'agent' || v === 'agent_occluded';

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
  /** A 2XLarge box on the event's own frame — the server stamps the frame. */
  | { mode: 'pick'; box: [number, number, number, number] }
  | { mode: 'occluded' }
  | { mode: 'auto' };

/** ByteTrack tracklets. */
export interface TrackData {
  tracklets: { rally_id: number; track_id: number; frames: number[]; boxes: [number, number, number, number][] }[];
}

/** Which tracklet each event's actor sits on (extraction/links.py). */
export type TrackLinks = Record<string, { rally_id: number; track_id: number }>;

/** GET /tracklets/masks — one rally's masks, whole tracklets at once.
 *  Values are base64 packed bits (box-crop space, ``mask_hw`` grid), row i
 *  aligned with the tracklet's i-th frame from /tracklets/{name}. */
export interface TrackMasks {
  mask_hw: [number, number];
  tracks: Record<string, string>;
}

/** Whether a human could endorse what was proposed here: an agent's
 *  proposal (lands as `confirmed` / `occluded`), or an automatic pick nobody
 *  has ruled on (lands as `confirmed`). */
export const canConfirm = (r: Pick<ReidRecord, 'resolution' | 'actor_review'>) => {
  const v = verdictOf(r);
  return v === 'agent' || v === 'agent_occluded' || (v === 'unreviewed' && r.resolution === 'auto');
};

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
  return rallyAt(rallies, event.time != null ? event.time : event.frame / fps);
};

/** The tracklet an event's actor sits on, as a stable "rally:track" key. */
export const trackKeyOf = (links: TrackLinks, id: string) => {
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


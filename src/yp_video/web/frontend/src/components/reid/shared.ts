/** Types and helpers shared by the ReID Label page, its video player and
 *  its group board. */

import { ApiError } from '@/lib/api';

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

/** {box} = manual pick, {none} = nobody is the actor, {} = revert to auto.
 *  ``frame`` marks a cross-frame pick: the box (and the crop the backend
 *  cuts) lives on that frame — the actor went undetected on the event's.
 *  ``snap: false`` forbids the backend's IoU snap onto a stored detection —
 *  sent when the clicked mask ruled that no detection is this player. */
export type ActorFix = { box?: [number, number, number, number]; none?: boolean; frame?: number; snap?: boolean };

/** ByteTrack tracklets + which tracklet each event's actor sits on. */
export interface TrackData {
  tracklets: { rally_id: number; track_id: number; frames: number[]; boxes: [number, number, number, number][] }[];
  links: Record<string, { rally_id: number; track_id: number }>;
}

/** GET /reid/track-masks — one rally's masks, whole tracklets at once.
 *  Values are base64 packed bits (box-crop space, ``mask_hw`` grid), row i
 *  aligned with the tracklet's i-th frame from /reid/tracks. */
export interface TrackMasks {
  mask_hw: [number, number];
  tracks: Record<string, string>;
}

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

/** Tracklet masks and boxes — the labeling overlay's pure core.
 *
 *  Nothing here touches React. Instance masks arrive as packed bits (see
 *  tracklets/store.py save_track_masks); decoding them, finding a tracklet's
 *  row at a frame and tinting it are pure functions of those bits and boxes,
 *  kept out of EventVideoPlayer so the fiddly bit indexing stays alone.
 */

import type { TrackData, TrackMasks } from './shared';

export type Box = [number, number, number, number];

/** One tracked player's box on one frame. */
export interface TrackBox {
  key: string;
  trackId: number;
  box: Box;
}

/** Decoded /tracklets/masks payload: key → concatenated packed rows, row i ↔
 *  the tracklet's i-th frame in the tracks jsonl. */
export interface MaskData {
  mh: number;
  mw: number;
  byKey: Map<string, Uint8Array>;
}

/** key → (frame → row index), the other half of a mask-row lookup. */
export type FrameRows = Map<string, Map<number, number>>;

/** One tracklet's mask at one frame, in its box's crop space. */
export interface Silhouette {
  key: string;
  box: Box;
  bits: Uint8Array;
  mw: number;
  mh: number;
  /** Row within the tracklet — the stable half of the render cache key. */
  row: number;
}

export interface RenderedSilhouette extends Silhouette {
  /** Tinted PNG data URL, transparent off-pixels. */
  url: string;
}

/** One bit of a packbits row, MSB first (matching numpy.packbits). */
export const bitAt = (bits: Uint8Array, i: number): boolean => Boolean((bits[i >> 3]! >> (7 - (i & 7))) & 1);

// Exact frame first, then ±1: a stride-decoded tracking run leaves gaps, and
// the playhead lands between detected frames. One policy, every lookup.
const NEAR_OFFSETS = [0, -1, 1];

/** A frame-keyed map's entry at (or immediately beside) a frame. */
export function nearestFrame<T>(map: Map<number, T>, frame: number): T | undefined {
  for (const off of NEAR_OFFSETS) {
    const hit = map.get(frame + off);
    if (hit !== undefined) return hit;
  }
  return undefined;
}

export function decodeMaskData(payload: TrackMasks | undefined): MaskData | null {
  if (!payload) return null;
  const byKey = new Map<string, Uint8Array>();
  for (const [key, b64] of Object.entries(payload.tracks)) {
    // A plain loop into a preallocated array, NOT Uint8Array.from(s, cb):
    // the callback form invokes JS once per byte and measured 146 ms on a
    // real rally (4.4 MB of packed masks) against 9 ms for this. Every rally
    // change pays this on the main thread, so the 16x matters.
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    byKey.set(key, bytes);
  }
  return { mh: payload.mask_hw[0], mw: payload.mask_hw[1], byKey };
}

export function buildFrameRows(tracklets: TrackData['tracklets']): FrameRows {
  const out: FrameRows = new Map();
  for (const t of tracklets) {
    const rows = new Map<number, number>();
    t.frames.forEach((f, i) => rows.set(f, i));
    out.set(`${t.rally_id}:${t.track_id}`, rows);
  }
  return out;
}

/** frame → the boxes of every tracklet detected there (~286k boxes on a real
 *  cut, built once per tracking payload). */
export function buildTrackBoxes(tracklets: TrackData['tracklets']): Map<number, TrackBox[]> {
  const map = new Map<number, TrackBox[]>();
  for (const t of tracklets) {
    const key = `${t.rally_id}:${t.track_id}`;
    t.frames.forEach((f, i) => {
      const box = t.boxes[i];
      if (!box) return;
      let arr = map.get(f);
      if (!arr) map.set(f, (arr = []));
      arr.push({ key, trackId: t.track_id, box });
    });
  }
  return map;
}

/** One tracklet's packed mask row at one exact frame. */
export function maskRowAt(
  maskData: MaskData,
  frameRows: FrameRows,
  key: string,
  frame: number,
): { bits: Uint8Array; row: number } | null {
  const row = frameRows.get(key)?.get(frame);
  const all = maskData.byKey.get(key);
  if (row === undefined || !all) return null;
  const rowBytes = (maskData.mh * maskData.mw) >> 3;
  return { bits: all.subarray(row * rowBytes, (row + 1) * rowBytes), row };
}

/** …at (or immediately beside) a frame — same policy as nearestFrame. */
export function maskRowNear(
  maskData: MaskData,
  frameRows: FrameRows,
  key: string,
  frame: number,
): { bits: Uint8Array; row: number } | null {
  for (const off of NEAR_OFFSETS) {
    const hit = maskRowAt(maskData, frameRows, key, frame + off);
    if (hit) return hit;
  }
  return null;
}

// A rally's worth of tinted silhouettes: ~12 players × a few hundred frames,
// and only the prev/next action's tracks ever take a second tint. Bounded so
// a long rally can't grow the cache without limit.
const MAX_CACHED_SILHOUETTES = 2048;

/** Tinted-silhouette data URLs, cached across frames.
 *
 *  A tracklet's mask row is immutable, so (track, row, tint) always yields
 *  the same PNG — without this every presented frame re-encodes one per
 *  player on screen, and toDataURL is not free. FIFO-evicted; clear() when
 *  the mask payload changes, the URLs belong to that rally's tracklets.
 */
export class SilhouetteRenderer {
  private cache = new Map<string, string>();

  url(sil: Silhouette, tint: string): string {
    const key = `${sil.key}|${sil.row}|${tint}`;
    const hit = this.cache.get(key);
    if (hit !== undefined) return hit;
    const url = tintedMaskUrl(sil, tint);
    if (this.cache.size >= MAX_CACHED_SILHOUETTES) {
      // Map iterates in insertion order — the oldest entry goes first.
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }
    this.cache.set(key, url);
    return url;
  }

  clear(): void {
    this.cache.clear();
  }
}

/** The mask painted in one flat color, off-pixels transparent. */
function tintedMaskUrl({ bits, mw, mh }: Silhouette, tint: string): string {
  const alpha = document.createElement('canvas');
  alpha.width = mw;
  alpha.height = mh;
  const alphaCtx = alpha.getContext('2d')!;
  const img = alphaCtx.createImageData(mw, mh);
  for (let i = 0; i < mw * mh; i++) {
    if (bitAt(bits, i)) img.data[i * 4 + 3] = 255;
  }
  alphaCtx.putImageData(img, 0, 0);

  const tinted = document.createElement('canvas');
  tinted.width = mw;
  tinted.height = mh;
  const ctx = tinted.getContext('2d')!;
  ctx.fillStyle = tint;
  ctx.fillRect(0, 0, mw, mh);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(alpha, 0, 0);
  return tinted.toDataURL();
}

/** Every silhouette on the current frame, tinted by ``tintOf`` and rendered
 *  through the cache. */
export function buildFrameSilhouettes(
  maskData: MaskData | null,
  frameRows: FrameRows,
  trackBoxes: Map<number, TrackBox[]>,
  frame: number,
  tintOf: (key: string) => string,
  renderer: SilhouetteRenderer,
): RenderedSilhouette[] {
  if (!maskData) return [];
  const list = nearestFrame(trackBoxes, frame);
  if (!list) return [];
  const out: RenderedSilhouette[] = [];
  for (const t of list) {
    const row = maskRowNear(maskData, frameRows, t.key, frame);
    if (!row) continue;
    const sil: Silhouette = {
      key: t.key,
      box: t.box,
      bits: row.bits,
      mw: maskData.mw,
      mh: maskData.mh,
      row: row.row,
    };
    out.push({ ...sil, url: renderer.url(sil, tintOf(t.key)) });
  }
  return out;
}

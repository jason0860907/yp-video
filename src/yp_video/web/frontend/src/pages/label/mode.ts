/** Contracts between the unified Label page and its four mode panels.
 *
 *  Each panel module exports a `ModeDescriptor` binding its status semantics
 *  (lib/labelStatus.ts, shared with the sidebar counters) to its UI: filter
 *  choices, availability, row extras, the Done endpoint.
 */

import type { ReactNode } from 'react';
import type { LabelMode, LabelStatus, UnionVideo } from '@/lib/labelStatus';

/** Shared Status-filter choices; a mode may append extras (e.g. re-pick). */
export const STATUS_OPTIONS: { value: string; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'unlabeled', label: 'Unlabeled' },
  { value: 'pre-annotate', label: 'Pre-Annotate' },
  { value: 'in-progress', label: 'In-Progress' },
  { value: 'done', label: 'Done' },
];

/** Which store a panel loads from. Only modes keeping more than one store
 *  offer the select (rally, action); association and reid have a single
 *  store — their machine output is merged into it (auto picks on extraction
 *  records) or derived on demand (reid clusters), never a separate file. */
export type LabelSource = 'auto' | 'annotation' | 'pre-annotation';

/** Which store actually satisfied the last load — what Auto resolved to.
 *  Rendered as a badge beside the Source select, so the editor always says
 *  whether you are looking at human labels or machine output. */
export type LoadedSource = 'annotation' | 'pre-annotation' | 'vlm' | 'none';

export interface ModeDescriptor {
  key: LabelMode;
  label: string;
  /** Choices for the page-level Status select while this mode is active. */
  statusOptions: { value: string; label: string }[];
  /** Where the row sits in the shared vocabulary while this mode is active. */
  status: (row: UnionVideo) => LabelStatus;
  /** Does the Status filter keep this row? 'all' keeps everything. */
  matches: (row: UnionVideo, status: string) => boolean;
  /** Can this mode's panel open the row's video? */
  available: (row: UnionVideo) => boolean;
  /** What the row is missing — the disabled tab's title. */
  hint: (row: UnionVideo) => string;
  /** Mode-specific chips rendered after the name in the combobox row, in
   *  addition to the shared status chip. */
  rowExtras?: (row: UnionVideo) => ReactNode;
  /** PUT endpoint toggling the stored Done flag, for the page-level Done
   *  button. Absent when the panel owns the button instead (ReID, whose
   *  Done also confirms auto actors and saves first). */
  doneApi?: (video: string) => string;
  /** react-query key of this mode's listing, refetched after Done toggles. */
  listKey: string;
  /** True when the mode keeps more than one store and the page should show
   *  the shared Source select (see LabelSource). */
  hasSources?: boolean;
}

/** Shared playhead across mode panels: a panel writes the position while its
 *  player runs and reads it back when it loads, so switching tabs resumes the
 *  same video at the same time. Keyed by video name — a different pick never
 *  inherits the previous video's position. */
export interface PlaybackClock {
  read(video: string): number | null;
  write(video: string, t: number): void;
}

/** Resolves true when leaving the current panel is allowed (the panel asked
 *  the user about unsaved work and they chose to discard it). */
export type DirtyGuard = () => boolean | Promise<boolean>;

/** Passed to the active panel; call with null on unmount to deregister. */
export type RegisterGuard = (guard: DirtyGuard | null) => void;

/** The labeling domain's shared vocabulary: the modes, the per-mode status
 *  of a video, and the union row both surfaces render.
 *
 *  Status itself is computed server-side where each work list is assembled
 *  (grep `"status":` in web/routers) — the same verdicts /label/stats counts
 *  for the sidebar, so no surface can disagree with another. The functions
 *  here only answer for a mode's absence on a union row. Lives in lib/
 *  because both a page and a layout component consume it — neither may
 *  import from the other.
 */

import type { ActionVideo, AssociationVideo, CutKind, LabelStatus, ReidVideo } from '@/types/api';

export type { LabelStatus };

export type LabelMode = 'rally' | 'action' | 'association' | 'reid';

/** One row of the annotate-results listing (rally annotation files). */
export interface RallyResult {
  name: string;
  source: string | string[];
  kind: string;
  /** The stored "rally labeling is finished" flag (core/label_done.py). */
  done: boolean;
  status: LabelStatus;
}

/** One row of the union video list, keyed by cut filename. Each mode's
 *  listing row is attached where it exists; absence means that mode has
 *  nothing for this video yet. */
export interface UnionVideo {
  /** Cut filename, extension included — the shared video identity. */
  name: string;
  kind: CutKind;
  /** Rally annotation exists in R2 but no cut is listed locally —
   *  only the Rally tab can open this row. */
  rallyOnly?: boolean;
  rally?: RallyResult;
  action?: ActionVideo;
  assoc?: AssociationVideo;
  reid?: ReidVideo;
}

export const rallyStatus = (row: UnionVideo): LabelStatus => row.rally?.status ?? 'unlabeled';
export const actionStatus = (row: UnionVideo): LabelStatus => row.action?.status ?? 'unlabeled';
export const assocStatus = (row: UnionVideo): LabelStatus => row.assoc?.status ?? 'unlabeled';
export const reidStatus = (row: UnionVideo): LabelStatus => row.reid?.status ?? 'unlabeled';

/** The one Status select, as video-picker filter options: the Label page's
 *  vocabulary over the server-computed `status` every work-list row carries.
 *  The predict pages spread these and append their operational extras
 *  (e.g. "No SPOT output"), mirroring how Label modes append theirs. */
export const STATUS_FILTER_OPTIONS: Array<{
  value: string;
  label: string;
  predicate: (row: { status: LabelStatus }) => boolean;
}> = [
  { value: 'all', label: 'All', predicate: () => true },
  { value: 'unlabeled', label: 'Unlabeled', predicate: (row) => row.status === 'unlabeled' },
  { value: 'pre-annotate', label: 'Pre-Annotate', predicate: (row) => row.status === 'pre-annotate' },
  { value: 'in-progress', label: 'In-Progress', predicate: (row) => row.status === 'in-progress' },
  { value: 'done', label: 'Done', predicate: (row) => row.status === 'done' },
];

/** The labeling domain's shared vocabulary: the modes, the per-mode status
 *  of a video, and the union row both surfaces render.
 *
 *  One place answers "is this video done / in-progress / pre-annotate" so
 *  the Label page's Status filters and the sidebar's pipeline counters can
 *  never disagree. Lives in lib/ because both a page and a layout component
 *  consume it — neither may import from the other.
 */

import type { ActionVideo, AssociationVideo, CutKind, ReidVideo } from '@/types/api';

export type LabelMode = 'rally' | 'action' | 'association' | 'reid';

/** The one status vocabulary every mode speaks.
 *
 *  unlabeled — nothing exists for this mode yet;
 *  pre-annotate — only machine output (a pre-label, an auto policy pass);
 *  in-progress — a human started but has not claimed to be finished;
 *  done — the human pressed Done (a stored flag, never derived from counts).
 */
export type LabelStatus = 'unlabeled' | 'pre-annotate' | 'in-progress' | 'done';

/** One row of the annotate-results listing (rally annotation files). */
export interface RallyResult {
  name: string;
  source: string | string[];
  kind: string;
  /** The stored "rally labeling is finished" flag (core/label_done.py). */
  done: boolean;
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

// The rally list endpoint returns source as an array of
// {annotation, spot-pre-annotation, pre-annotation}; a file counts as
// "labeled" once a manual annotation exists for it.
const sources = (src: string | string[]) => (Array.isArray(src) ? src : [src]);
const hasAnnotation = (src: string | string[]) => sources(src).includes('annotation');

export const rallyStatus = (row: UnionVideo): LabelStatus => {
  if (!row.rally) return 'unlabeled';
  if (row.rally.done) return 'done';
  return hasAnnotation(row.rally.source) ? 'in-progress' : 'pre-annotate';
};

export const actionStatus = (row: UnionVideo): LabelStatus => {
  const v = row.action;
  if (!v) return 'unlabeled';
  if (v.done) return 'done';
  // A human file exists (provenance by store, like rally); done stays a
  // separate, explicit claim — saving alone keeps the video In-Progress.
  if (v.has_action_final_annotation) return 'in-progress';
  return v.has_action_pre_annotation ? 'pre-annotate' : 'unlabeled';
};

export const assocStatus = (row: UnionVideo): LabelStatus => {
  const v = row.assoc;
  if (!v) return 'unlabeled';
  if (v.done) return 'done';
  if (v.reviewed > 0) return 'in-progress';
  // The row exists at all only once extraction ran, and the automatic
  // policy's picks ARE machine pre-annotation awaiting review.
  return 'pre-annotate';
};

export const reidStatus = (row: UnionVideo): LabelStatus => {
  const v = row.reid;
  if (!v) return 'unlabeled';
  if (v.done) return 'done';
  if ((v.player_count ?? 0) > 0) return 'in-progress';
  // Embeddings are the machine's prep work — computed but nobody grouped.
  return v.embedded_models.length > 0 ? 'pre-annotate' : 'unlabeled';
};

const MODE_STATUS: Record<LabelMode, (row: UnionVideo) => LabelStatus> = {
  rally: rallyStatus,
  action: actionStatus,
  association: assocStatus,
  reid: reidStatus,
};

export type StatusCounts = Record<LabelStatus, number>;

/** Per-mode tally of the union list — videos, not events. */
export function countLabelStatuses(videos: UnionVideo[]): Record<LabelMode, StatusCounts> {
  const zero = (): StatusCounts => ({ unlabeled: 0, 'pre-annotate': 0, 'in-progress': 0, done: 0 });
  const counts = Object.fromEntries(
    Object.keys(MODE_STATUS).map((mode) => [mode, zero()]),
  ) as Record<LabelMode, StatusCounts>;
  for (const row of videos) {
    for (const [mode, status] of Object.entries(MODE_STATUS)) {
      counts[mode as LabelMode][status(row)] += 1;
    }
  }
  return counts;
}

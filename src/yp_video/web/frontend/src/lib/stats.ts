import type { SystemStats } from '@/types/api';

/**
 * The pipeline counters, in pipeline order — one row per stage that produces
 * something durable on disk. Shared by the sidebar footer and the Jobs page so
 * the two cannot disagree about what "progress" means.
 *
 * Every value counts VIDEOS, not events: `/api/system/stats` globs one file
 * per stem in each stage's directory.
 */
export const STAT_ROWS: Array<[label: string, key: keyof SystemStats]> = [
  ['Videos', 'videos'],
  ['Cuts', 'cuts'],
  ['Rally Labels', 'annotations'],
  ['Action Pred', 'action_pre_annotations'],
  ['Action Labels', 'actions'],
  ['Association Labels', 'association_labels'],
  ['ReID Labels', 'reid_labels'],
];

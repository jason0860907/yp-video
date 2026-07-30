import type { SystemStats } from '@/types/api';

/**
 * The pipeline counters, in pipeline order — one row per stage that produces
 * something durable on disk. Shared by the sidebar footer and the Jobs page so
 * the two cannot disagree about what "progress" means.
 *
 * Every value counts VIDEOS, not events. Association is the one progress
 * value: it is rendered as Done / Started, where Started includes In Progress.
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

/** Association progress is displayed as Done / Started (including In Progress). */
export function formatStatValue(
  stats: SystemStats | undefined,
  key: keyof SystemStats,
): number | string {
  if (key === 'association_labels') {
    return `${stats?.association_labels_done ?? 0}/${stats?.association_labels ?? 0}`;
  }
  return stats?.[key] ?? 0;
}

/** Display names for the metrics in the common SPOT task contract, shared by
 *  the task table, the per-task epoch charts and the breakdown panel. */
export const METRIC_LABELS: Record<string, string> = {
  harmonic_mAP: 'Harmonic mAP',
  segment_mAP: 'Segment mAP',
  temporal_mAP: 'Temporal mAP',
  spatial_mAP: 'Spatial mAP',
  winner_top1: 'Winner Top-1',
  majority_baseline: 'Majority baseline',
  loss: 'Loss',
};

export const TASK_LABELS: Record<string, string> = {
  rally: 'Rally',
  winner: 'Winner',
  action: 'Action',
  location: 'Location',
};

/** Fixed task order so every run lists and colors its tasks the same way. */
export const TASK_ORDER = ['action', 'rally', 'winner', 'location'] as const;

/** Display names for the metrics in the common SPOT task contract, shared by
 *  the task table and the per-task epoch charts. */
export const METRIC_LABELS: Record<string, string> = {
  harmonic_mAP: 'Harmonic mAP',
  temporal_mAP: 'Temporal mAP',
  spatial_mAP: 'Spatial mAP',
  overall_top1: 'Overall Top-1',
  player_top1: 'Player Top-1',
  occluded_recall: 'Occluded recall',
  untracked_recall: 'Untracked recall',
  loss: 'Loss',
};

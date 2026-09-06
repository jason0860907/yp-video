/** Display names for the metrics in the common SPOT task contract, shared by
 *  the task table, the per-task epoch charts and the breakdown panel. */
export const METRIC_LABELS: Record<string, string> = {
  harmonic_mAP: 'Harmonic mAP',
  segment_mAP: 'Segment mAP',
  temporal_mAP: 'Temporal mAP',
  spatial_mAP: 'Spatial mAP',
  overall_top1: 'Overall Top-1',
  player_top1: 'Player Top-1',
  winner_top1: 'Winner Top-1',
  majority_baseline: 'Majority baseline',
  occluded_recall: 'Occluded recall',
  untracked_recall: 'Untracked recall',
  loss: 'Loss',
};

export const TASK_LABELS: Record<string, string> = {
  rally: 'Rally',
  winner: 'Winner',
  action: 'Action',
  location: 'Location',
  actor: 'Actor',
};

/** Fixed task order so every run lists and colors its tasks the same way. */
export const TASK_ORDER = ['action', 'rally', 'winner', 'actor', 'location'] as const;

/** Actor target kinds of the contract (yp_spot.contract.ACTOR_TARGET_KINDS). */
export const ACTOR_KIND_LABELS: Record<string, string> = {
  track: 'Tracked player',
  occluded: 'Occluded',
  untracked: 'Untracked',
};

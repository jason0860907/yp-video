/** Shared backend response shapes. Hand-written for the routes the UI reads;
 *  the data-contract record types live in src/types/contracts (generated). */

export type JobStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'pending' | 'stopped';

export interface JobItem {
  status?: string;
  video?: string;
  progress?: number;
  message?: string;
  error?: string;
  /** Unix seconds — stamped when the item starts / settles. */
  started_at?: number;
  finished_at?: number;
}

export interface Job {
  id: string;
  status: JobStatus;
  name?: string;
  type?: string;
  progress?: number;
  message?: string;
  error?: string;
  /** Number of retained log lines; fetch bodies from /jobs/{id}/logs. */
  log_count?: number;
  /** Unix seconds. `started_at` is set on the first transition to running. */
  created_at?: number;
  started_at?: number | null;
  // items is the batch sub-progress; other keys are job-type-specific payloads.
  params?: { items?: JobItem[]; [k: string]: unknown };
}

export interface JobLogs {
  lines: string[];
  count: number;
}

export interface ActionTrainStatus {
  active_job?: Job;
  spot_available?: boolean;
  init_checkpoints?: Array<{ value: string; label: string }>;
  resumable_runs?: Array<{ value: string; label: string }>;
  action_annotations?: {
    videos?: number;
    events?: number;
    frames?: number;
    label_dir?: string;
    frame_dir?: string;
    checkpoint_dir?: string;
    by_view?: Record<'broadcast' | 'sideline', { videos?: number; events?: number; frames?: number }>;
    per_video?: Array<{ video: string; events: number; frames: number; view: string; is_val?: boolean }>;
  };
  vnl_1_5?: {
    ready?: boolean;
    train_videos?: number;
    train_events?: number;
    val_videos?: number;
    val_events?: number;
    frame_dir?: string;
    frame_dir_exists?: boolean;
  };
}

export interface ActionVideoMap {
  video: string;
  harmonic: number;
  temporal: number;
  spatial: number;
  events: number;
}

export interface ActionMapBreakdown {
  temporal: { tolerances: number[]; classes: Record<string, number[]>; overall: number[] };
  spatial: { pixel_tolerances: number[]; overall_by_px: number[]; overall: number };
  per_video?: ActionVideoMap[];
}

export interface ActionPerfEntry {
  epoch: number;
  lr?: number | null;
  val_mAP?: number;
  val_mAP_temporal?: number;
  val_mAP_spatial?: number;
  train_loss?: number | null;
  val_loss?: number | null;
  per_class?: Record<string, number>;
  val_per_video?: ActionVideoMap[] | null;
}

export interface ActionPerfData {
  run?: string;
  meta?: Record<string, unknown> | null;
  best?: { epoch?: number; value?: number } | null;
  entries: ActionPerfEntry[];
  runs?: string[];
}

/** Either breakdown flavour; discriminate with `'spatial' in bd`. */
export type MapBreakdown = ActionMapBreakdown | RallyMapBreakdown;

/** Live training progress — job.params.{action,rally}_train_progress. */
export interface TrainProgress {
  epoch?: number;
  epoch_display?: number;
  epochs?: number;
  phase?: string;
  phase_label?: string;
  phase_progress?: number;
  step?: number;
  total?: number;
  current_loss?: number;
  latest_train_loss?: number;
  latest_val_loss?: number;
  latest_val_map?: number;
  latest_val_breakdown?: MapBreakdown;
  best_value?: number;
  best_epoch?: number;
  best_breakdown?: MapBreakdown;
}

/** SPOT rally (segment) training — /spot-train/status. */
export interface RallyTrainStatus {
  active_job?: Job;
  spot_available?: boolean;
  init_checkpoints?: Array<{ value: string; label: string }>;
  resumable_runs?: Array<{ value: string; label: string }>;
  rally_annotations?: {
    videos?: number;
    rallies?: number;
    rally_hours?: number;
    total_hours?: number;
    with_local_video?: number;
    missing_videos?: number;
    label_dir?: string;
  };
  frame_caches?: Array<{ fps: string; videos: number }>;
  rally_checkpoints?: { dir?: string; runs?: number; exists?: boolean };
}

/** Segment-mAP breakdown (per class per tIoU); no spatial component. */
export interface RallyMapBreakdown {
  temporal: { tolerances: number[]; classes: Record<string, number[]>; overall: number[] };
  per_video?: ActionVideoMap[];
}

/** Video record from the SPOT rally predict listing. */
export interface RallyPredictVideo {
  name: string;
  kind: CutKind;
  has_annotation?: boolean;
  /** SPOT prediction exists (rally-spot-pre-annotations). */
  has_pre_annotation?: boolean;
  /** VLM prediction exists (rally-pre-annotations) — a separate file. */
  has_vlm_pre_annotation?: boolean;
}

export interface VllmStatus {
  status: 'running' | 'starting' | 'stopped' | 'error';
  model?: string;
  port?: number;
  max_num_seqs?: number;
}

export type CutKind = 'broadcast' | 'sideline';

export interface VideoMeta {
  name: string;
  kind: CutKind;
  has_detection?: boolean;
}

/** Video record from the action-annotate listing (richer than VideoMeta). */
export interface ActionVideo {
  name: string;
  kind: CutKind;
  event_count?: number;
  has_action_annotation?: boolean;
  has_action_pre_annotation?: boolean;
  has_action_final_annotation?: boolean;
  action_reviewed?: boolean;
  action_annotation_source?: string;
  rally_sources?: string[];
}

/** Action-label editor data (one video's rallies + action events). */
export interface ActionRally {
  rally_id: number;
  start: number;
  end: number;
  label?: string;
}
export interface ActionEvent {
  id: string;
  rally_id: number | null;
  frame: number;
  time: number | null;
  relative_frame: number | null;
  label: string;
  xy: [number, number];
  visible: boolean;
}
export interface ActionAnnotationData {
  video?: string;
  source_video?: string;
  duration?: number;
  fps?: number;
  num_frames?: number;
  rallies?: Array<{ rally_id?: unknown; start?: unknown; end?: unknown; label?: string }>;
  events?: Array<Record<string, unknown>>;
}

/** Decoded audio envelope for the Action Label waveform lane. */
export interface WaveformData {
  video: string;
  loading: boolean;
  error: string;
  hasAudio: boolean;
  duration: number;
  peaks: number[];
  rms: number[];
}

export interface SpotCheckpoint {
  path: string;
  name: string;
  epoch?: number;
  is_best?: boolean;
  best_metric?: string | null;
  best_value?: number | null;
  size_mb?: number;
}

export interface SpotInfo {
  available: boolean;
  spot_dir?: string;
  checkpoints?: SpotCheckpoint[];
  default_checkpoint?: string;
  error?: string;
}

/** How far a video has walked the stage chain (see extraction/prerequisites.py).
 *  `blocked_on` is the FIRST unmet stage; later gaps are its consequences. */
export interface PipelineState {
  /** Rally source tags present, in priority order (manual → SPOT → VLM). */
  rally_sources: string[];
  has_action: boolean;
  has_tracks: boolean;
  /** Tracking ran before instance masks existed — the picker loses silhouette
   *  arbitration but labels are unaffected. */
  has_masks: boolean;
  /** The rallies moved since these tracklets were cut, so every
   *  "{rally_id}:{track_id}" key now points somewhere else. */
  tracks_stale: boolean;
  has_records: boolean;
  blocked_on: 'rallies' | 'action' | 'tracks' | 'records' | null;
}

/** Video record from the reid listing. */
export interface ReidVideo {
  name: string;
  kind: CutKind;
  event_count: number;
  has_reid: boolean;
  reid_counts?: { ok: number; multi: number; miss: number } | null;
  /** Embedders whose matrix exists for this video (backfill fills gaps). */
  embedded_models: string[];
  /** Existing matrices being refreshed after an actor fix. */
  stale_embedding_models?: string[];
  /** Distinct saved identities (0 = extracted but not labeled yet). */
  player_count?: number;
  /** Labeling marked finished by the user (the Label page's Done button). */
  done?: boolean;
  pipeline: PipelineState;
}

/** GET /reid/options — the server-side model registry. */
export interface ReidOptions {
  keypoint_sources: string[];
  default_embedder: string;
  embedders: {
    name: string;
    threshold: { min: number; max: number; default: number; step: number };
    /** Embeds background-suppressed crops — the viewer should show those. */
    masked: boolean;
  }[];
  /** Official clip-reident default first, then candidates; Predict/Backfill
   *  can explicitly override the active package. */
  checkpoints: { ref: string; run_name: string; active: boolean }[];
}

/** One action event's extraction outcome (embedding stripped server-side). */
export interface ReidRecord {
  id: string;
  frame: number;
  time?: number | null;
  label?: string;
  /** Contact point (normalized); null for invisible / point-less events —
   *  those never auto-associate and are assigned via (cross-frame) picks. */
  xy: [number, number] | null;
  /** false = the action isn't visible on its frame. */
  visible?: boolean;
  /** Set when the crop was cut from another frame (cross-frame pick). */
  crop_frame?: number;
  /** ok = unique person box, multi = ranked pick among overlaps, miss = none. */
  status: 'ok' | 'multi' | 'miss';
  /** How the actor was resolved. Always explicit — never infer it from crop
   *  presence, the server never does either. */
  resolution: 'unresolved' | 'auto' | 'manual' | 'occluded';
  /** The human verdict on this event's actor; drives association training. */
  actor_review?: 'unreviewed' | 'confirmed_auto' | 'manual' | 'occluded';
  /** What the rule decided, plus the learned ranker's opinion when a
   *  shadow is activated. The rule is what produced the crop. */
  association?: {
    version: string;
    decision: 'selected' | 'ambiguous' | 'no_candidate';
    candidate_count: number;
    margin: number | null;
    confidence?: number | null;
    none_probability?: number | null;
    top: {
      box: [number, number, number, number];
      cost: number;
      source: 'wrist' | 'box' | 'other';
      detection_score: number;
    } | null;
    learned?: {
      version: string;
      decision: 'selected' | 'ambiguous' | 'no_candidate';
      candidate_count: number;
      margin: number | null;
      confidence?: number | null;
      none_probability?: number | null;
      top: {
        box: [number, number, number, number];
        cost: number;
        source: 'wrist' | 'box' | 'other';
        detection_score: number;
      } | null;
    } | null;
  };
  box?: [number, number, number, number] | null;
  score?: number | null;
  candidates: number;
  crop?: string | null;
  /** 17 COCO keypoints of the matched player, crop-relative [x, y, conf]. */
  keypoints?: [number, number, number][] | null;
  /** ALL person detections on the event frame — the actor picker's choices. */
  detections?: { box: [number, number, number, number]; score: number }[];
  /** The automatic pick, kept when a manual fix overrides it. */
  auto_box?: [number, number, number, number] | null;
}

export interface ReidActorFixResponse {
  record: ReidRecord;
  track_link: { rally_id: number; track_id: number } | null;
  /** Non-visible models refreshing after the response. */
  refreshing_models: string[];
}

export interface ReidCluster {
  id: number;
  size: number;
  unit_keys: string[];
}

/** GET /reid/clusters — units and how they group.
 *
 *  A UNIT is what identity is about: the tracklet a crop belongs to
 *  ("t:<rally>:<track>"), or the event itself when no tracklet reaches it
 *  ("e:<event_id>"). `units` carries each one's events so the board can draw
 *  crops without a second round trip. */
export interface ReidClusters {
  threshold: number;
  model: string;
  units: Record<string, { event_ids: string[] }>;
  clusters: ReidCluster[];
}

/** Saved identities + nearest-centroid matches for one video.
 *
 *  Names live on the unit: `tracks` names a tracklet (and with it every
 *  action it performed), `assignments` names a single event — for events no
 *  tracklet reaches, and for the ones that contradict their tracklet, which
 *  is what an identity switch looks like. An event name WINS over its
 *  tracklet's. `unit_names` is the two resolved onto units. */
export interface ReidPlayers {
  tracks: Record<string, string>;
  assignments: Record<string, string>;
  unit_names: Record<string, string>;
  players: string[];
  /** Keyed by UNIT key, not event id. */
  matches: Record<string, { player: string; sim: number; assigned: boolean }>;
}

export interface SystemStats {
  videos?: number;
  cuts?: number;
  pre_annotations?: number;
  annotations?: number;
  action_pre_annotations?: number;
  actions?: number;
  association_labels?: number;
  reid_labels?: number;
}

export interface ActiveCount {
  count: number;
}

// ── ReID Train ──
/** One recording session: the videos sharing a player name-space
 *  (reid/sessions.py infers this from shared assigned names). */
export interface ReidSession {
  id: string;
  stems: string[];
  players: string[];
  counts: Record<string, number>;
  /** Merge evidence: player name -> the stems carrying it. */
  shared: Record<string, string[]>;
  n_assigned: number;
  /** No name links this video to any other — usually inconsistent labeling. */
  is_isolated: boolean;
  models: Record<string, string[]>;
}

export interface ReidSlider {
  min: number;
  max: number;
  default: number;
  step: number;
}

/** A calibrated clustering cutoff, in cosine distance — tuned against the
 *  quality of the groups it produces, not against pairwise separability. */
export interface ReidThresholdSuggestion {
  suggested: number;
  /** Adjusted Rand index at `suggested` (1.0 = groups match the labels). */
  ari: number;
  /** Groups produced, against the true player count. More is expected. */
  n_clusters: number;
  n_ids: number;
  /** The whole sweep, for the chart. */
  curve: Array<{ t: number; ari: number; n: number }>;
  /** Pairwise separability, independent of any cutoff. Chance = 0.5. */
  auc: number;
  same_p50: number;
  same_p95: number;
  diff_p05: number;
  diff_p50: number;
  n_pos: number;
  n_neg: number;
  slider: ReidSlider;
}

export interface ReidScores {
  m_ap: number;
  rank1: number;
  rank5: number;
  n_query: number;
}

/** One video, scored on its own labels — the labeling unit. */
export interface ReidVideoEval {
  stem: string;
  model: string;
  n_ids: number;
  n_crops: number;
  n_assigned: number;
  coverage: number;
  dropped_singletons: number;
  dropped_unembedded: number;
  scores: ReidScores;
  threshold: ReidThresholdSuggestion;
}

/** Does an identity survive into another recording of the same session?
 *  Only players named on both sides can be scored; `n_skipped` are the ones
 *  simply not on court for the other clip. */
export interface ReidCrossEval {
  session_id: string;
  query_stem: string;
  gallery_stems: string[];
  n_ids_shared: number;
  n_scored: number;
  n_skipped: number;
  scores: ReidScores;
}

export interface ReidModelEval {
  model: string;
  crop_weighted?: ReidScores;
  macro?: ReidScores;
  totals?: { n_videos: number; n_ids: number; n_crops: number; coverage: number };
  threshold?: ReidThresholdSuggestion;
  current_threshold: ReidSlider;
  /** Videos this model has no embeddings for. */
  skipped: string[];
  videos: ReidVideoEval[];
  cross_video: ReidCrossEval[];
}

export interface ReidPerfData {
  models: ReidModelEval[];
  evaluated_at: number;
}

/** GET /actor-association/videos — one row of the Association work list. */
export interface AssociationVideo {
  name: string;
  kind: 'broadcast' | 'sideline';
  /** Records extraction wrote — the review denominator. */
  event_count: number;
  reviewed: number;
  unreviewed: number;
  /** verdict -> count; keys are absent when the count is zero. */
  verdicts: Partial<Record<'manual' | 'occluded' | 'confirmed_auto', number>>;
  /** What the automatic policy produced, for context on the remainder. */
  auto_counts: { ok: number; multi: number; miss: number };
  pipeline: PipelineState;
}

export interface ReidAssociationMetrics {
  reviewed: number;
  positive: number;
  occluded: number;
  candidate_recall: number | null;
  top1_accuracy: number | null;
  auto_coverage: number | null;
  selective_accuracy: number | null;
  operational_precision?: number | null;
  occluded_rejection_rate: number | null;
  overall_accuracy: number | null;
  threshold?: number;
}

export interface ReidAssociationPerfData {
  models: Record<string, ReidAssociationMetrics>;
  candidates?: Record<string, ReidAssociationMetrics | null>;
  labels: Record<string, number>;
  skipped: Record<string, number>;
  dataset?: ReidAssociationDatasetSummary;
}

export interface ReidAssociationDatasetSummary {
  examples: number;
  stems: number;
  labels: Record<string, number>;
  skipped: Record<string, number>;
}

export interface ReidAssociationCheckpoint {
  name: string;
  active_shadow: boolean;
  /** Which feature contract it was trained on: 'box-v2' | 'track-v1'. */
  feature_set: string;
  /** Why it cannot be the extraction shadow, or null when it can. */
  shadow_blocked_on: string | null;
  created_at: number;
  threshold: number;
  metrics: {
    grouped_oof?: ReidAssociationMetrics;
  };
  training: {
    examples: number;
    stems: string[];
  };
}

/** A yp-spot checkpoint carrying the actor head — it answers "who acted" by
 *  looking at the frames, so it has holdout metrics rather than grouped-OOF
 *  ones and is selected through `spot_checkpoint`. */
export interface SpotActorCheckpoint {
  path: string;
  name: string;
  epoch: number | null;
  mtime: number | null;
  holdout: string | null;
  metrics: {
    all_top1?: number;
    hard_top1?: number;
    manual_top1?: number;
    rule_manual_top1?: number;
  };
  note: string | null;
}

export interface ReidAssociationStatus {
  dataset: ReidAssociationDatasetSummary;
  checkpoints: ReidAssociationCheckpoint[];
  spot_checkpoints: SpotActorCheckpoint[];
  active_shadow: string | null;
  active_job: Job | null;
}

export interface ReidDatasetInfo {
  name: string;
  created_at: number;
  counts: Record<string, number>;
  config: Record<string, unknown>;
}

/** One yp-reid checkpoint package (reid/checkpoints/<run>/). */
export interface ReidRun {
  path: string;
  run_name: string;
  /** True only for the fixed production default, never inferred from metrics. */
  active: boolean;
  source: 'trained' | 'imported' | null;
  architecture: string | null;
  embedding_dim: number | null;
  best_metric: string | null;
  best_value: number | null;
  metrics: { m_ap?: number; rank1?: number; rank5?: number; n_query?: number } | null;
  created_at: string | null;
  note: string | null;
  mtime: number;
}

export interface ReidTrainStatus {
  sessions: ReidSession[];
  models: Array<{ name: string; labeled_videos: number; threshold: ReidSlider }>;
  /** Everything is scoped to finished (Done-marked) videos; pending_videos
   *  counts labeled-but-unfinished cuts that are excluded from training. */
  totals: { ready_videos: number; pending_videos: number; assigned_events: number; identities: number; sessions: number };
  datasets: ReidDatasetInfo[];
  split_modes: string[];
  reid_engine_available: boolean;
  runs: ReidRun[];
  active_job: Job | null;
}

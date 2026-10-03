/** Shared backend response shapes. Hand-written for the routes the UI reads;
 *  the data-contract record types live in src/types/contracts (generated). */

/** The one label-status vocabulary every mode speaks, computed server-side
 *  where each work list is assembled (grep `"status":` in web/routers).
 *
 *  unlabeled — nothing exists for this mode yet;
 *  pre-annotate — only machine output (a pre-label, an auto policy pass);
 *  in-progress — a human started but has not claimed to be finished;
 *  done — the human pressed Done (a stored flag, never derived from counts).
 */
export type LabelStatus = 'unlabeled' | 'pre-annotate' | 'in-progress' | 'done';

/** GET /label/stats — per-mode status tally of the union video list,
 *  keys in pipeline order (web/routers/label_stats.py). */
export type LabelStats = Record<
  'rally' | 'action' | 'association' | 'reid' | 'detection' | 'court',
  Record<LabelStatus, number>
>;

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
  /** Cloudflare Access identity of whoever started the run. */
  actor?: string;
}

/** One row of the audit trail (GET /audit/events).
 *
 *  `action` is the FastAPI route template ("POST /api/annotate/annotations")
 *  or a job transition ("job.completed") — see lib/auditLabels.ts for the
 *  display names. `repeats` counts autosaves folded into this row. */
export interface AuditEvent {
  id: number;
  /** Session start — the first save folded into this row. ISO-8601, UTC. */
  first_at: string;
  /** Session end — the last save folded into this row. ISO-8601, UTC. */
  at: string;
  actor: string;
  action: string;
  target: string | null;
  summary: Record<string, unknown>;
  outcome: 'ok' | 'error';
  status: number | null;
  duration_ms: number | null;
  repeats: number;
}

export interface AuditPage {
  events: AuditEvent[];
  /** Cursor for the next page, or null at the end of the trail. */
  next_before: number | null;
}

/** GET /audit/worklog — labeling time per person over a range. */
export interface WorklogSession {
  /** First and last save of an unbroken run of work (ISO timestamps). */
  start: string;
  end: string;
  saves: number;
}

export interface WorklogPerson {
  actor: string;
  /** Longest-first across people; chronological within a person. */
  sessions: WorklogSession[];
}

export interface Worklog {
  since: string;
  until: string;
  people: WorklogPerson[];
}

/** One item changed by a save: added / removed / edited, or a truncation
 *  marker when a single save changed more items than are worth listing. */
export interface AuditChange {
  op: 'added' | 'removed' | 'edited' | 'truncated';
  id?: string | number;
  /** The whole item, for added and removed. */
  item?: Record<string, unknown>;
  /** Only the fields that moved: field -> [before, after]. */
  fields?: Record<string, [unknown, unknown]>;
  /** How many further changes were not listed (op: 'truncated'). */
  count?: number;
}

export interface AuditSave {
  /** ISO-8601. */
  at: string;
  changes: AuditChange[];
}

/** GET /audit/events/{id}/saves — every save folded into one row. */
export interface AuditSaves {
  /** Oldest first. */
  saves: AuditSave[];
}

export interface AuditFilters {
  actors: string[];
  actions: string[];
  /** Video names / categories already present in the trail. */
  targets: string[];
}

export interface Me {
  email: string;
}

export interface JobLogs {
  lines: string[];
  count: number;
}

export interface ActionAnnotationStats {
  videos?: number;
  events?: number;
  frames?: number;
  label_dir?: string;
  frame_dir?: string;
  checkpoint_dir?: string;
  by_view?: Record<'broadcast' | 'sideline', { videos?: number; events?: number; frames?: number }>;
  per_video?: Array<{
    video: string;
    events: number;
    frames: number;
    view: string;
    is_val?: boolean;
  }>;
}

export type FusionRecipeId =
  | 'rally'
  | 'rally_winner'
  | 'action'
  | 'action_rally_winner';
export type SpotTask = 'rally' | 'winner' | 'action' | 'location' | 'person';

/** One entry of the contract's task-set registry (yp_video.contracts.action.RECIPES). */
export interface FusionRecipe {
  id: FusionRecipeId;
  name: string;
  tasks: SpotTask[];
  description: string;
  /** Request fields the form shows for this recipe, on top of the common ones. */
  fields: string[];
  /** Request values the form resets to when this recipe is picked. */
  defaults: Record<string, unknown>;
  serveable_tasks: SpotTask[];
}

export interface FusionModelStatus {
  recipes: FusionRecipe[];
  task_labels: Record<string, string>;
  spot_available: boolean;
  /** Per recipe: packages carrying every head the recipe trains. */
  init_checkpoints: Record<string, SelectOption[]>;
  action_annotations?: ActionAnnotationStats;
  rally_annotations?: {
    videos?: number;
    rallies?: number;
    rally_hours?: number;
    total_hours?: number;
    with_video?: number;
    missing_videos?: number;
    per_video?: Array<{ video: string; view: string; is_val?: boolean }>;
  };
  active_job: Job | null;
}

/** Point-spotting (action) and segment (rally) detail: AP per class per
 *  temporal tolerance (frames, or tIoU for segments), the overall row, and
 *  per-video temporal mAP. */
export interface SpottingBreakdown {
  tolerances: number[];
  classes: Record<string, number[]>;
  overall: number[];
  per_video: Array<{ video: string; temporal: number; events: number }>;
}

/** Contact-point detail: mAP per pixel tolerance and per-video spatial mAP. */
export interface LocationBreakdown {
  pixel_tolerances: number[];
  overall_by_px: number[];
  per_video: Array<{ video: string; spatial: number; events: number }>;
}

/** Court-side confusion: rows are ground truth, columns predictions, both in
 *  `classes` order; `recall` is the diagonal over the row total. */
export interface WinnerBreakdown {
  classes: string[];
  confusion: number[][];
  recall: Record<string, number | null>;
}

export type TaskBreakdown = SpottingBreakdown | LocationBreakdown | WinnerBreakdown;

export interface TaskMetricPhase {
  loss: number | null;
  /** Scalar quality measures only; structured detail lives in `breakdown`. */
  metrics: Record<string, number | null>;
  counts: Record<string, number>;
  /** Task-specific detail, null when the phase was not evaluated. */
  breakdown: TaskBreakdown | null;
}

/** Common contract emitted by every enabled head in a SPOT training run. */
export interface TaskMetricSnapshot {
  primary_metric: string;
  train: TaskMetricPhase;
  validation: TaskMetricPhase;
}

export type TaskMetrics = Record<string, TaskMetricSnapshot>;

/** One epoch line of yp-spot's metrics.jsonl. */
export interface TrainEpochRecord {
  epoch: number;
  lr: number | null;
  loss: { train: number | null; val: number | null };
  tasks: TaskMetrics;
  /** The checkpoint criterion this epoch was ranked by. */
  selection: { task: string; metric: string; mode: 'min' | 'max'; value: number };
  best: boolean;
}

export interface TrainPerfData {
  run?: string;
  meta?: Record<string, unknown> | null;
  best?: { epoch?: number; value?: number } | null;
  entries: TrainEpochRecord[];
  runs?: string[];
}

/** Live training progress — job.params.{action,rally,association}_train_progress. */
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
  latest_task_metrics?: TaskMetrics;
  best_value?: number;
  best_epoch?: number;
  best_task_metrics?: TaskMetrics;
}

/** Video record from the SPOT rally predict listing. */
export interface RallyPredictVideo {
  name: string;
  kind: CutKind;
  status: LabelStatus;
  has_annotation?: boolean;
  /** SPOT prediction exists (rally-spot-pre-annotations). */
  has_pre_annotation?: boolean;
}

/** One row of the Inference page: which stage outputs the cut already has. */
export interface InferenceVideo {
  name: string;
  kind: CutKind;
  /** SPOT rally pre-annotation exists (rally-spot/pre-annotations). */
  has_rally_spot: boolean;
  /** Machine action pre-annotation exists (action/pre-annotations). */
  has_action_pre: boolean;
  /** Tracklets exist AND were cut against the video's current rallies; an
   *  Inference run keeps tracking only when this holds for its tracker. */
  tracks_current: boolean;
  /** Which tracker cut those tracks; null when there are none. */
  tracker: Tracker | null;
  pipeline: PipelineState;
}

export type CutKind = 'broadcast' | 'sideline';

/** The association step that linked detections into tracklets. */
export type Tracker = 'bytetrack' | 'mcbyte';

/** A <select> option as the backend serves it (checkpoints). */
export interface SelectOption {
  value: string;
  label: string;
}

/** Video record from the action-annotate listing. */
export interface ActionVideo {
  name: string;
  kind: CutKind;
  event_count?: number;
  /** Something is loadable — the human file, or the machine one. */
  has_action_annotation?: boolean;
  /** Only machine output exists; provenance is by store, not a field. */
  has_action_pre_annotation?: boolean;
  /** A human-saved annotation exists (the human-only annotations dir). */
  has_action_final_annotation?: boolean;
  /** The stored "action labeling is finished" flag (core/label_done.py) —
   *  independent of saving, which only writes the human store. */
  done?: boolean;
  status: LabelStatus;
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
  /** Which store this payload came from (null = neither exists yet); the
   *  file's own provenance stays in its `source` metadata. */
  loaded_source?: 'annotation' | 'pre-annotation' | null;
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
  /** Heads the package carries (manifest.tasks). */
  tasks?: SpotTask[];
  recipe?: FusionRecipeId | null;
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
  /** Rally source tags present, in priority order (manual → SPOT). */
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

/** GET /extraction/videos — the player-detection work list. */
export interface ExtractionVideo {
  name: string;
  kind: CutKind;
  event_count: number;
  has_records: boolean;
  /** People found across every action frame. What was DECIDED about them is
   *  the association listing's to report. */
  detections?: number | null;
  /** Detector identifier persisted in the record header. */
  detector?: string | null;
  /** Tracklets exist, carry masks, and were cut against the video's current
   *  rallies. Tracking without Overwrite redoes everything else. */
  tracks_current: boolean;
  /** Which tracker cut those tracks; null when there are none. */
  tracker: Tracker | null;
  pipeline: PipelineState;
}

/** GET /reid/videos — how far a video's player naming has got. */
export interface ReidVideo {
  name: string;
  kind: CutKind;
  event_count: number;
  /** Embedders whose matrix exists for this video. */
  embedded_models: string[];
  /** Existing matrices being refreshed after an actor fix. */
  stale_embedding_models?: string[];
  /** Distinct saved identities (0 = extracted but not labeled yet). */
  player_count?: number;
  /** Labeling marked finished by the user (the Label page's Done button). */
  done?: boolean;
  status: LabelStatus;
  pipeline: PipelineState;
}

/** GET /reid/options — the embedder registry. */
export interface ReidOptions {
  default_embedder: string;
  embedders: {
    name: string;
    threshold: { min: number; max: number; default: number; step: number };
    /** Embeds background-suppressed crops — the viewer should show those. */
    masked: boolean;
  }[];
  /** Official clip-reident default first, then candidates; the embed job can
   *  explicitly override the active package. */
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
  /** Present (true) only when the verdict names a person but resolves to no
   *  tracklet today (links.unresolved_labels) — tracklet training skips such
   *  an event until the player is re-picked or tracking improves. */
  actor_review_unresolved?: boolean;
  /** What decided this actor: the policy (`source`) and whether it found a
   *  candidate (`status`). */
  association?: {
    source: string;
    status: 'selected' | 'no_candidate';
  };
  box?: [number, number, number, number] | null;
  score?: number | null;
  candidates: number;
  crop?: string | null;
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
 *  Names are stored per event (`assignments`); tracklet ids do not survive a
 *  re-track. `unit_names` names each unit whose named events agree — a unit
 *  whose events disagree (an identity switch) stays unnamed. */
export interface ReidPlayers {
  assignments: Record<string, string>;
  unit_names: Record<string, string>;
  players: string[];
  /** Keyed by UNIT key, not event id. */
  matches: Record<string, { player: string; sim: number; assigned: boolean }>;
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

/** GET /actor-association/videos — one row of the Association work list.
 *  Only videos carrying everything association is built on are listed:
 *  rallies, action labels and extraction records. */
export interface AssociationVideo {
  name: string;
  kind: 'broadcast' | 'sideline';
  /** The review denominator: current action events inside a rally that name
   *  somebody to identify (a `score` names nobody). Action annotations own
   *  this, not extraction — see extraction/store.labelable_actions. */
  event_count: number;
  reviewed: number;
  unreviewed: number;
  /** verdict -> count; keys are absent when the count is zero. */
  verdicts: Partial<Record<'manual' | 'occluded' | 'confirmed_auto', number>>;
  /** Reviewed verdicts resolving to no tracklet today — need re-picking (or
   *  a better tracking run) before tracklet training can use them. Zero when
   *  the video has no tracking run at all: that gap is the pipeline chip's. */
  unresolved: number;
  /** What the automatic policy produced, for context on the remainder. */
  auto_counts: { ok: number; multi: number; miss: number };
  /** The stored "actor review is finished" flag — a human verdict, no longer
   *  derived from the counts above. */
  done: boolean;
  status: LabelStatus;
  pipeline: PipelineState;
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

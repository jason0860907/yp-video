/** Shared backend response shapes. Hand-written for the routes the UI reads;
 *  the data-contract record types live in src/types/contracts (generated). */

export type JobStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'pending' | 'stopped';

export interface JobItem {
  status?: string;
  video?: string;
  progress?: number;
  message?: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  name?: string;
  type?: string;
  progress?: number;
  message?: string;
  error?: string;
  logs?: string[];
  // items is the batch sub-progress; other keys are job-type-specific payloads.
  params?: { items?: JobItem[]; [k: string]: unknown };
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

export interface ActionTrainProgress {
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
  latest_val_breakdown?: ActionMapBreakdown;
  best_value?: number;
  best_epoch?: number;
  best_breakdown?: ActionMapBreakdown;
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

/** Video record from the TAD predict listing. `features` is keyed by feature
 *  model (base/large/giant/gigantic) → present. */
export interface PredictVideo {
  name: string;
  kind: CutKind;
  has_annotation?: boolean;
  has_pre_annotation?: boolean;
  has_prediction?: boolean;
  features?: Record<string, boolean>;
}

export interface TrainCheckpoint {
  path: string;
  name: string;
  size_mb: number;
  kind: 'best' | 'last' | 'epoch';
}

/** Train (TAD) page status + video records + perf chart shapes. */
export interface TrainStatus {
  cuts_count?: number;
  features_by_model?: Record<string, number>;
  annotations_exist?: boolean;
  vllm_running?: boolean;
  active_train_job?: Job;
}

export interface TrainVideo {
  name: string;
  kind: CutKind;
  has_annotation?: boolean;
  has_pre_annotation?: boolean;
  has_features?: boolean;
  has_prediction?: boolean;
}

export interface TrainConfigDefaults {
  lr?: number;
  epochs?: number;
  warmup_epochs?: number;
  weight_decay?: number;
  schedule?: string;
  batch_size?: number;
  sampler_alpha?: number;
}

export interface PerfEntry {
  epoch: number;
  tiou?: Record<string, { mAP?: number }>;
  per_source?: Record<string, { mAP?: number; tiou_mAP?: number[]; n_videos?: number; n_preds?: number }>;
}
export interface PerfData {
  name?: string;
  entries?: PerfEntry[];
}

export interface SystemStats {
  videos?: number;
  cuts?: number;
  pre_annotations?: number;
  annotations?: number;
  action_pre_annotations?: number;
  actions?: number;
  vjepa_b?: number;
  predictions?: number;
}

export interface ActiveCount {
  count: number;
}

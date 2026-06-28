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
  action_annotations?: { videos?: number; events?: number; frames?: number; label_dir?: string };
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

export interface ActionTrainProgress {
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
  best_value?: number;
  best_epoch?: number;
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

export interface SpotCheckpoint {
  path: string;
  name: string;
  epoch?: number;
  is_best?: boolean;
}

export interface SpotInfo {
  available: boolean;
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

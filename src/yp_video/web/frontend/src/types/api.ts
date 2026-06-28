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
  params?: { items?: JobItem[] };
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

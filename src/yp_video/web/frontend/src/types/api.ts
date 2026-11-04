/** Backend response shapes, named for the components that read them.
 *
 *  Generated: every alias below points into src/types/openapi.ts, which
 *  `make api-types` regenerates from the routes' response models — change a
 *  shape in the backend, never here. The few hand-written types left are
 *  either client-side state or payloads whose schema another repo owns
 *  (yp-spot's metrics records) or that is still open on the backend
 *  (extraction records). */
import type { components } from './openapi';

type S = components['schemas'];

export type LabelStatus = S['ActionVideo']['status'];
export type LabelStats = S['LabelStats'];
export type JobStatus = S['JobStatus'];

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
/** `params` is job-type-specific on the backend (`dict`); the batch
 *  sub-progress under `items` is the one part every job page reads. */
export type Job = Omit<S['JobSummary'], 'params'> & {
  params: { items?: JobItem[]; [k: string]: unknown };
};
export type AuditEvent = S['AuditEvent'];
export type AuditPage = S['AuditPage'];
export type WorklogSession = S['WorklogSession'];
export type WorklogPerson = S['WorklogPerson'];
export type Worklog = S['Worklog'];
export type AuditChange = S['AuditChange'];
export type AuditSave = S['AuditSave'];
export type AuditSaves = S['AuditSaves'];
export type AuditFilters = S['AuditFilters'];
export type Me = S['Me'];
export type JobLogs = S['JobLogs'];
export type ActionAnnotationStats = S['ActionAnnotationStats'];
export type FusionRecipeId = S['FusionRecipe']['id'];
export type SpotTask = S['FusionRecipe']['tasks'][number];
export type FusionRecipe = S['FusionRecipe'];
export type FusionModelStatus = S['FusionModelStatus'];

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

/** Class confusion (winner / actor side / jump): rows are ground truth,
 *  columns predictions, both in `classes` order; `recall` is the diagonal
 *  over the row total. */
export interface ConfusionBreakdown {
  classes: string[];
  confusion: number[][];
  recall: Record<string, number | null>;
}

/** One event-metric summary of the actor head (yp-spot actor_training.EventMetrics). */
export interface ActorEventSummary {
  events: number;
  proposal_hit: number | null;
  actor_hit: number | null;
  class_hit: number | null;
  joint_hit: number | null;
  by_class?: Record<string, Omit<ActorEventSummary, 'by_class'>>;
}

/** The actor head over held-out events, picking among the annotated boxes. */
export interface ActorBreakdown {
  annotated_proposals: ActorEventSummary;
}

export type TaskBreakdown = SpottingBreakdown | LocationBreakdown | ConfusionBreakdown | ActorBreakdown;

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
/** The run's metrics.jsonl records are yp-spot's (open on this backend). */
export type TrainPerfData = Omit<S['TrainPerfData'], 'entries' | 'best'> & {
  entries: TrainEpochRecord[];
  best?: { epoch?: number; value?: number } | null;
};

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
export type InferenceVideo = S['InferenceVideo'];
export type CutKind = S['ActionVideo']['kind'];
export type SelectOption = S['SelectOption'];
export type ActionVideo = S['ActionVideo'];
export type CourtSide = S['CourtSide'];
export type ActionRally = S['ActionRally'];
/** An event as the action editor holds it: the served event
 *  (`NormalizedActionEvent`) with `time` derived client-side and an unset
 *  side / jump left absent rather than null. */
export type ActionEvent = Omit<S['NormalizedActionEvent'], 'time' | 'score' | 'side' | 'jump'> & {
  time: number | null;
  side?: CourtSide;
  jump?: boolean;
};
export type ActionAttributeDefaults = S['ActionAttributeDefaults'];
export type ActionAnnotationData = S['ActionAnnotationData'];

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
export type SpotCheckpoint = S['SpotCheckpoint'];
export type SpotInfo = S['SpotInfo'];
export type PipelineState = S['PipelineState'];
export type ExtractionVideo = S['ExtractionVideo'];
export type ReidVideo = S['ReidVideo'];
export type ReidOptions = S['ReidOptions'];

/** One action event's extraction outcome (embedding stripped server-side). */
export interface ReidRecord {
  id: string;
  frame: number;
  time?: number | null;
  label?: string;
  /** Contact point (normalized); null for invisible / point-less events —
   *  those never auto-associate and are assigned by hand. */
  xy: [number, number] | null;
  /** false = the action isn't visible on its frame. */
  visible?: boolean;
  /** ok = unique person box, multi = ranked pick among overlaps, miss = none. */
  status: 'ok' | 'multi' | 'miss';
  /** How the actor was resolved. Always explicit — never infer it from crop
   *  presence, the server never does either. */
  resolution: 'unresolved' | 'auto' | 'manual' | 'occluded';
  /** The verdict on this event's actor (actor/labels.py): a human's —
   *  association training truth — or an agent's proposal awaiting Confirm. */
  actor_review?: 'unreviewed' | 'manual' | 'confirmed' | 'occluded' | 'agent' | 'agent_occluded';
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
}
/** `record` is the stored extraction record, still open on the backend. */
export type ReidActorFixResponse = Omit<S['ReidActorFixResponse'], 'record'> & { record: ReidRecord };
export type ReidCluster = S['ReidCluster'];
export type ReidClusters = S['ReidClusters'];
export type ReidPlayers = S['ReidPlayers'];
export type ActiveCount = S['ActiveCount'];
export type ReidSession = S['ReidSession'];
export type ReidSlider = S['ReidSlider'];
export type ReidThresholdSuggestion = S['ReidThresholdSuggestion'];
export type ReidScores = S['ReidScores'];
export type ReidVideoEval = S['ReidVideoEval'];
export type ReidCrossEval = S['ReidCrossEval'];
export type ReidModelEval = S['ReidModelEval'];
export type ReidPerfData = S['ReidPerfData'];
export type AssociationVideo = S['AssociationVideo'];
export type BoxCheckStatus = NonNullable<S['BoxCheckEntry']['status']>;
export type BoxCheckEntry = S['BoxCheckEntry'];
export type ReidDatasetInfo = S['ReidDatasetInfo'];
/** `metrics` is the checkpoint manifest's open dict; these are the keys the
 *  trainer writes. */
export type ReidRun = Omit<S['ReidRun'], 'metrics'> & {
  metrics: { m_ap?: number; rank1?: number; rank5?: number; n_query?: number } | null;
};
export type ReidTrainStatus = S['ReidTrainStatus'];

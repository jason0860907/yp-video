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

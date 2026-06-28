import type { Job } from '@/types/api';

/** A job is settled once it reaches a terminal state. */
export const isTerminal = (status: Job['status']): boolean =>
  status === 'completed' || status === 'failed' || status === 'cancelled';

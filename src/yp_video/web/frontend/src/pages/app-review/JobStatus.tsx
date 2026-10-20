/** Where the worker is with a match: the newest analysis, identification and
 *  court job, each as one badge — progress while it runs, the error once it
 *  fails. */

import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { formatDateTime, formatDuration } from '@/lib/format';
import type { JobKind, LibraryMatch, MatchJob } from './types';

const TONE: Record<MatchJob['status'], BadgeTone> = {
  pending: 'warning',
  running: 'info',
  completed: 'success',
  failed: 'danger',
  superseded: 'neutral',
};

const KINDS: JobKind[] = ['analysis', 'identify', 'court'];

const jobLabel = (kind: JobKind, job?: MatchJob) =>
  kind === 'analysis' ? 'Analysis' : kind === 'court' ? 'Court 3D' : job?.mode === 'advanced' ? 'Advanced ID' : 'Identify';

function title(job: MatchJob): string {
  const run = job.started_at && job.finished_at ? ` · ran ${formatDuration(job.finished_at - job.started_at)}` : '';
  return `${job.status} · created ${formatDateTime(job.created_at)}${run}${job.error ? `\n${job.error}` : ''}`;
}

export function JobBadge({ kind, job }: { kind: JobKind; job?: MatchJob }) {
  if (!job) return <span className="text-text-muted">—</span>;
  const active = job.status === 'pending' || job.status === 'running';
  return (
    <span className="inline-flex flex-col items-start gap-0.5" title={title(job)}>
      <Badge tone={TONE[job.status]} dot={job.status === 'running'}>
        {kind === 'identify' && job.mode === 'advanced' ? 'advanced · ' : ''}
        {job.status}
        {active && ` ${Math.round(job.progress)}%`}
      </Badge>
      {active && job.step && <span className="text-[11px] text-text-muted">{job.step}</span>}
      {job.status === 'failed' && (
        <span className="text-[11px] text-red-400">{job.error_code ?? 'failed'}</span>
      )}
    </span>
  );
}

/** All three as a labelled row, for the match header. */
export function JobRow({ match }: { match: LibraryMatch }) {
  return (
    <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {KINDS.map((kind) => (
        <span key={kind} className="inline-flex items-center gap-1.5">
          {jobLabel(kind, match.jobs[kind])}
          <JobBadge kind={kind} job={match.jobs[kind]} />
        </span>
      ))}
    </span>
  );
}

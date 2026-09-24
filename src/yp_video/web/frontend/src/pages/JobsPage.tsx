import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { ProgressBar } from '@/components/job/ProgressBar';
import { JobItems } from '@/components/job/JobItems';
import { formatClock } from '@/lib/format';
import { statusLabel, statusTheme } from '@/lib/job';
import { LabelProgress } from '@/components/labeling/LabelProgress';
import { toast } from '@/components/feedback/toast';
import type { Job } from '@/types/api';

const POLL_MS = 15_000;


export function JobsPage() {
  const qc = useQueryClient();

  const jobs = useQuery({
    queryKey: ['jobs-list'],
    queryFn: () => apiFetch<Job[]>(API.jobs.list),
    refetchInterval: POLL_MS,
  });

  const refetchJobs = () => qc.invalidateQueries({ queryKey: ['jobs-list'] });

  const cancelJob = async (id: string) => {
    try {
      await apiFetch(API.jobs.cancel(id), { method: 'POST' });
      toast.warning('Job cancelled');
      refetchJobs();
    } catch (e) {
      toast.error(`Cancel failed: ${errMsg(e)}`);
    }
  };

  const allJobs = jobs.data ?? [];
  const running = allJobs.filter((j) => j.status === 'running');
  const completed = allJobs.filter((j) => j.status === 'completed').length;
  const failed = allJobs.filter((j) => j.status === 'failed').length;

  const sortedJobs = [...allJobs].sort((a, b) => {
    if (a.status === 'running' && b.status !== 'running') return -1;
    if (b.status === 'running' && a.status !== 'running') return 1;
    return (b.id || '').localeCompare(a.id || '');
  });

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <Button size="sm" onClick={refetchJobs}>
            Refresh
          </Button>
        }
      />

      <div className="grid grid-cols-3 gap-3.5">
        <StatTile label="Active jobs" value={running.length} tintClass="text-primary-light" sub="running now" />
        <StatTile label="Completed" value={completed} tintClass="text-primary-light" />
        <StatTile label="Failed" value={failed} tintClass={failed ? 'text-red-400' : 'text-text-muted'} />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {/* Job queue */}
        <Card>
          <SectionLabel>Job queue</SectionLabel>
          {sortedJobs.length === 0 ? (
            <EmptyState
              icon={
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              }
              title="No jobs"
              subtitle="Jobs appear when you start detection, training, or inference"
            />
          ) : (
            <div className="space-y-2.5">
              {sortedJobs.map((job) => (
                <JobRow key={job.id} job={job} onCancel={cancelJob} />
              ))}
            </div>
          )}
        </Card>

        <Card>
          <SectionLabel>Label progress</SectionLabel>
          <LabelProgress />
        </Card>
      </div>
    </div>
  );
}

function JobRow({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const isRunning = job.status === 'running';

  // One structure for every status — header, bar (while running), message —
  // so a job doesn't change shape as it moves through its lifecycle, and a
  // growing message can never crowd the bar or change any element's width.
  return (
    <div className="rounded-xl border border-border bg-surface-50 px-3.5 py-3">
      <div className="flex items-center gap-3.5">
        <span className={cn('h-2 w-2 flex-shrink-0 rounded-full', statusTheme(job.status).dot, isRunning && 'animate-pulse-dot')} />
        <div className="flex min-w-0 flex-1 items-center gap-2.5">
          <span className="truncate text-[12.5px] font-medium text-text-primary">{job.name || job.type || 'unknown'}</span>
          {job.type && (
            <span className="flex-shrink-0 rounded bg-ink/5 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-text-muted">
              {job.type}
            </span>
          )}
        </div>
        {(job.started_at ?? job.created_at) != null && (
          <span className="hidden flex-shrink-0 font-mono text-[10px] tabular-nums text-text-muted sm:inline">
            {formatClock(job.started_at ?? job.created_at)}
          </span>
        )}
        <span className={cn('flex-shrink-0 text-right font-mono text-[11px] font-medium tabular-nums', statusTheme(job.status).text)}>
          {statusLabel(job)}
        </span>
        {isRunning && (
          <button
            type="button"
            onClick={() => onCancel(job.id)}
            className="flex-shrink-0 rounded-lg px-2 py-1 text-[11px] font-medium text-red-400/80 transition-colors hover:bg-red-500/10 hover:text-red-300"
          >
            Cancel
          </button>
        )}
      </div>
      {isRunning && (
        <div className="mt-2">
          <ProgressBar progress={job.progress} />
        </div>
      )}
      {job.message && (
        <div className="mt-1.5 truncate font-mono text-[10.5px] text-text-muted" title={job.message}>
          {job.message}
        </div>
      )}
      {job.error && (
        <p className="mt-1.5 truncate text-[11px] text-red-400/80" title={job.error}>
          {job.error}
        </p>
      )}
      <JobItems items={job.params?.items ?? []} maxVisible={16} />
    </div>
  );
}

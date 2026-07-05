import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { StatusBadge } from '@/components/job/StatusBadge';
import { ProgressBar } from '@/components/job/ProgressBar';
import { JobItems } from '@/components/job/JobItems';
import { toast } from '@/components/feedback/toast';
import type { Job, SystemStats, VllmStatus } from '@/types/api';

const POLL_MS = 15_000;

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

const DOT_CLS: Record<string, string> = {
  running: 'bg-primary-light',
  completed: 'bg-primary-light',
  failed: 'bg-red-400',
  cancelled: 'bg-amber-400',
  pending: 'bg-text-muted',
  stopped: 'bg-text-muted',
};

const STAT_ROWS: Array<[label: string, key: keyof SystemStats]> = [
  ['Videos', 'videos'],
  ['Cuts', 'cuts'],
  ['Rally-Pred', 'pre_annotations'],
  ['Rally Labels', 'annotations'],
  ['Action-Pred', 'action_pre_annotations'],
  ['Action Labels', 'actions'],
  ['VJEPA-B', 'vjepa_b'],
  ['TAD-Pred', 'predictions'],
];

export function JobsPage() {
  const qc = useQueryClient();

  const vllm = useQuery({
    queryKey: ['vllm-status'],
    queryFn: () => apiFetch<VllmStatus>(API.system.vllmStatus),
    refetchInterval: POLL_MS,
  });
  const stats = useQuery({
    queryKey: ['system-stats'],
    queryFn: () => apiFetch<SystemStats>(API.system.stats),
    refetchInterval: POLL_MS,
  });
  const jobs = useQuery({
    queryKey: ['jobs-list'],
    queryFn: () => apiFetch<Job[]>(API.jobs.list),
    refetchInterval: POLL_MS,
  });

  const refetchVllm = () => qc.invalidateQueries({ queryKey: ['vllm-status'] });
  const refetchJobs = () => qc.invalidateQueries({ queryKey: ['jobs-list'] });

  const startVllm = async () => {
    try {
      toast.info('Starting vLLM server…');
      await apiFetch(API.system.vllmStart, { method: 'POST' });
      toast.success('vLLM starting');
      setTimeout(refetchVllm, 3000);
    } catch (e) {
      toast.error(`Failed: ${errMsg(e)}`);
    }
  };
  const stopVllm = async () => {
    try {
      await apiFetch(API.system.vllmStop, { method: 'POST' });
      toast.success('vLLM stopped');
      refetchVllm();
    } catch (e) {
      toast.error(`Failed: ${errMsg(e)}`);
    }
  };
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

  const vllmStatus = vllm.data?.status ?? 'stopped';
  const vllmRunning = vllmStatus === 'running';

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <Button size="sm" onClick={refetchJobs}>
            Refresh
          </Button>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Active jobs" value={running.length} tintClass="text-primary-light" sub="running now" />
        <StatTile label="Completed" value={completed} tintClass="text-primary-light" />
        <StatTile label="Failed" value={failed} tintClass={failed ? 'text-red-400' : 'text-text-muted'} />
        <StatTile
          label="vLLM"
          value={vllmStatus}
          tintClass={vllmRunning ? 'text-primary-light' : 'text-text-muted'}
          sub={vllm.data?.model}
        />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1.6fr_1fr]">
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

        {/* System panel */}
        <div className="space-y-4">
          <Card>
            <SectionLabel>vLLM server</SectionLabel>
            <div className="mb-3 flex items-center gap-2.5">
              <StatusBadge status={vllmStatus} />
              <span className="font-mono text-[11px] text-text-muted">
                {vllm.data?.model ? `${vllm.data.model}` : 'no model'}
              </span>
            </div>
            <div className="mb-4 grid grid-cols-2 gap-2 text-[11px]">
              <div className="rounded-lg border border-border bg-surface-50 px-3 py-2">
                <div className="text-text-muted">Port</div>
                <div className="font-mono tabular-nums text-text-secondary">{vllm.data?.port ?? '—'}</div>
              </div>
              <div className="rounded-lg border border-border bg-surface-50 px-3 py-2">
                <div className="text-text-muted">Max seqs</div>
                <div className="font-mono tabular-nums text-text-secondary">{vllm.data?.max_num_seqs ?? '—'}</div>
              </div>
            </div>
            {vllmRunning ? (
              <Button intent="danger" size="sm" onClick={stopVllm} className="w-full">
                Stop server
              </Button>
            ) : (
              <Button intent="primary" size="sm" onClick={startVllm} className="w-full">
                Start server
              </Button>
            )}
          </Card>

          <Card>
            <SectionLabel>Corpus</SectionLabel>
            <div className="space-y-1.5 text-[11.5px]">
              {STAT_ROWS.map(([label, key]) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="text-text-secondary">{label}</span>
                  <span className="font-mono tabular-nums text-text-primary">{stats.data?.[key] ?? 0}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function JobRow({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const isRunning = job.status === 'running';
  const pct = Math.round((job.progress ?? 0) * 100);

  return (
    <div className="rounded-xl border border-border bg-surface-50 px-3.5 py-3">
      <div className="flex items-center gap-3.5">
        <span className={cn('h-2 w-2 flex-shrink-0 rounded-full', DOT_CLS[job.status] ?? DOT_CLS.pending, isRunning && 'animate-pulse-dot')} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-2.5">
            <span className="truncate text-[12.5px] font-medium text-text-primary">{job.name || job.type || 'unknown'}</span>
            {job.type && (
              <span className="flex-shrink-0 rounded bg-ink/5 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-text-muted">
                {job.type}
              </span>
            )}
          </div>
          {job.message && <div className="mt-0.5 truncate font-mono text-[10.5px] text-text-muted">{job.message}</div>}
        </div>
        {isRunning && (
          <div className="hidden w-36 flex-shrink-0 sm:block">
            <ProgressBar progress={job.progress} />
          </div>
        )}
        <span className="w-9 flex-shrink-0 text-right font-mono text-[11px] tabular-nums text-text-secondary">
          {isRunning ? `${pct}%` : '—'}
        </span>
        {isRunning ? (
          <button
            type="button"
            onClick={() => onCancel(job.id)}
            className="flex-shrink-0 rounded-lg px-2 py-1 text-[11px] font-medium text-red-400/80 transition-colors hover:bg-red-500/10 hover:text-red-300"
          >
            Cancel
          </button>
        ) : (
          <span className="w-[60px] flex-shrink-0 text-right text-[10px] uppercase text-text-muted">{job.status}</span>
        )}
      </div>
      {job.error && <p className="mt-2 truncate text-[11px] text-red-400/80">{job.error}</p>}
      <JobItems items={job.params?.items ?? []} maxVisible={16} />
    </div>
  );
}

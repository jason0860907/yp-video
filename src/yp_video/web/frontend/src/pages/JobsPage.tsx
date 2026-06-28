import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { StatusBadge } from '@/components/job/StatusBadge';
import { ProgressBar } from '@/components/job/ProgressBar';
import { JobItems } from '@/components/job/JobItems';
import { toast } from '@/components/feedback/toast';
import type { Job, VllmStatus } from '@/types/api';

const POLL_MS = 15_000;

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

export function JobsPage() {
  const qc = useQueryClient();

  const vllm = useQuery({
    queryKey: ['vllm-status'],
    queryFn: () => apiFetch<VllmStatus>(API.system.vllmStatus),
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

  const status = vllm.data?.status ?? 'stopped';
  const running = status === 'running';
  const vllmInfo = running
    ? `${vllm.data?.model ?? '—'} · port ${vllm.data?.port ?? '—'}`
    : vllm.data?.model
      ? `Model: ${vllm.data.model}`
      : 'Server stopped';

  const sortedJobs = [...(jobs.data ?? [])].sort((a, b) => {
    if (a.status === 'running' && b.status !== 'running') return -1;
    if (b.status === 'running' && a.status !== 'running') return 1;
    return (b.id || '').localeCompare(a.id || '');
  });

  return (
    <div className="mx-auto max-w-screen-2xl space-y-6">
      <PageHeader title="Jobs & System" subtitle="Monitor tasks and control vLLM server" />

      {/* vLLM control */}
      <Card>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-primary/20 bg-primary/10">
              <svg className="h-[18px] w-[18px] text-primary-light" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7m0 0a3 3 0 01-3 3" />
              </svg>
            </div>
            <div>
              <h3 className="font-heading text-sm font-semibold text-text-primary">vLLM Server</h3>
              <span className="text-[11px] leading-tight text-text-muted">{vllmInfo}</span>
            </div>
            <span className="ml-1">
              <StatusBadge status={status} />
            </span>
          </div>
          {running ? (
            <Button intent="danger" size="sm" onClick={stopVllm}>
              Stop
            </Button>
          ) : (
            <Button intent="primary" size="sm" onClick={startVllm}>
              Start
            </Button>
          )}
        </div>
      </Card>

      {/* Job list */}
      <Card>
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-heading text-sm font-semibold text-text-primary">Background Jobs</h3>
          <Button size="sm" onClick={refetchJobs}>
            Refresh
          </Button>
        </div>

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
            {sortedJobs.map((job) => {
              const isRunning = job.status === 'running';
              return (
                <div
                  key={job.id}
                  className={cn(
                    'group space-y-3 rounded-xl border p-4 transition-all duration-200',
                    isRunning
                      ? 'border-primary/20 bg-primary/[0.06] hover:border-primary/30 hover:bg-primary/[0.09]'
                      : 'border-border bg-surface-50/50 hover:border-border-light hover:bg-white/[0.03]',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <StatusBadge status={job.status} />
                      <span className="font-heading text-sm font-medium text-text-primary">
                        {job.name || job.type || 'unknown'}
                      </span>
                      <span className="text-[11px] text-text-muted opacity-60">{job.type}</span>
                    </div>
                    {isRunning && (
                      <button
                        type="button"
                        onClick={() => cancelJob(job.id)}
                        className="rounded-lg px-2 py-1 text-[11px] font-medium text-red-400/80 transition-all duration-200 hover:bg-red-500/10 hover:text-red-300"
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                  {isRunning && <ProgressBar progress={job.progress} />}
                  {job.message && <p className="truncate text-[11px] leading-relaxed text-text-muted">{job.message}</p>}
                  {job.error && <p className="truncate text-[11px] leading-relaxed text-red-400/80">{job.error}</p>}
                  <JobItems items={job.params?.items ?? []} maxVisible={16} />
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
}

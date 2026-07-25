import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import { toast } from '@/components/feedback/toast';
import { JobProgress } from '@/components/job/JobProgress';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { isTerminal } from '@/lib/job';
import { useSSE } from '@/lib/useSSE';
import type {
  Job,
  ReidAssociationMetrics,
  ReidAssociationPerfData,
  ReidAssociationStatus,
} from '@/types/api';

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const errMsg = (error: unknown) =>
  error instanceof ApiError
    ? error.body
    : error instanceof Error
      ? error.message
      : String(error);

export function AssociationTrainingCard() {
  const queryClient = useQueryClient();
  const [job, setJob] = useState<Job | null>(null);
  const statusQuery = useQuery({
    queryKey: ['actor-association-status'],
    queryFn: () =>
      apiFetch<ReidAssociationStatus>(API.association.status),
    refetchInterval: job && !isTerminal(job.status) ? false : 20_000,
  });
  const performanceQuery = useQuery({
    queryKey: ['actor-association-performance'],
    queryFn: () =>
      apiFetch<ReidAssociationPerfData>(
        API.association.performance,
      ),
    staleTime: 60_000,
  });
  const status = statusQuery.data;

  useEffect(() => {
    if (status?.active_job && !job) setJob(status.active_job);
  }, [job, status?.active_job]);

  useSSE<Job>(
    job && !isTerminal(job.status)
      ? API.jobs.eventsSSE(job.id)
      : null,
    (next) => {
      setJob(next);
      if (isTerminal(next.status)) {
        if (next.status === 'completed') {
          toast.success(next.message || 'Association training finished');
        } else if (next.status === 'failed') {
          toast.error(
            `Association training failed: ${next.error ?? 'unknown error'}`,
          );
        }
        void queryClient.invalidateQueries({
          queryKey: ['actor-association-status'],
        });
        void queryClient.invalidateQueries({
          queryKey: ['actor-association-performance'],
        });
      }
    },
  );

  const rows = useMemo(() => {
    const ruleRows = Object.entries(
      performanceQuery.data?.models ?? {},
    );
    const candidateRows = Object.entries(
      performanceQuery.data?.candidates ?? {},
    ).filter(
      (row): row is [string, ReidAssociationMetrics] => row[1] != null,
    );
    return [
      ...ruleRows.map(
        ([name, metrics]) =>
          [name, metrics, 'full labels'] as const,
      ),
      ...candidateRows.map(
        ([name, metrics]) =>
          [`learned:${name}`, metrics, 'grouped OOF'] as const,
      ),
    ];
  }, [performanceQuery.data]);

  const startTraining = async () => {
    try {
      const started = await apiFetch<Job>(
        API.association.train,
        {
          method: 'POST',
          body: {},
        },
      );
      setJob(started);
      toast.success('Association training started');
    } catch (error) {
      toast.error(`Training failed to start: ${errMsg(error)}`);
    }
  };

  const setShadow = async (checkpoint: string | null) => {
    try {
      await apiFetch(API.association.shadow, {
        method: 'PUT',
        body: { checkpoint },
      });
      toast.success(
        checkpoint
          ? `${checkpoint} enabled in shadow mode`
          : 'Learned shadow disabled',
      );
      void statusQuery.refetch();
    } catch (error) {
      toast.error(`Shadow update failed: ${errMsg(error)}`);
    }
  };

  const busy = Boolean(job && !isTerminal(job.status));
  const canTrain =
    (status?.dataset.examples ?? 0) >= 20
    && (status?.dataset.stems ?? 0) >= 2;

  return (
    <Card>
      <SectionLabel>Actor association · learned shadow</SectionLabel>
      <p className="mb-3 text-[11px] text-text-muted">
        A candidate scorer and an explicit NONE scorer train together.
        Validation holds out whole videos. Training never changes production
        or shadow activation automatically; the rule remains production.
      </p>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
        <span>{status?.dataset.examples ?? '—'} reviewed examples</span>
        <span>·</span>
        <span>{status?.dataset.stems ?? '—'} videos</span>
        <span>·</span>
        <span>
          shadow{' '}
          <span className="font-mono text-text-secondary">
            {status?.active_shadow ?? 'disabled'}
          </span>
        </span>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        <Button
          intent="primary"
          disabled={busy || !canTrain}
          onClick={() => void startTraining()}
        >
          {busy ? 'Training…' : 'Train candidate'}
        </Button>
        {status?.active_shadow ? (
          <Button onClick={() => void setShadow(null)}>
            Disable learned shadow
          </Button>
        ) : null}
        {!canTrain ? (
          <span className="self-center text-[11px] text-amber-400">
            Needs at least 20 explicit reviews across two videos.
          </span>
        ) : null}
      </div>

      {status?.checkpoints.length ? (
        <div className="mb-4 space-y-1">
          {status.checkpoints.map((checkpoint) => (
            <div
              key={checkpoint.name}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-50 px-2.5 py-1.5 text-[11px]"
            >
              <span className="font-mono text-text-primary">
                {checkpoint.name}
              </span>
              <span className="text-text-muted">
                {checkpoint.feature_set} · {checkpoint.training.examples}{' '}
                examples · threshold {checkpoint.threshold.toFixed(3)}
              </span>
              <Button
                className="ml-auto"
                disabled={
                  checkpoint.active_shadow ||
                  checkpoint.shadow_blocked_on !== null
                }
                title={checkpoint.shadow_blocked_on ?? undefined}
                onClick={() => void setShadow(checkpoint.name)}
              >
                {checkpoint.active_shadow
                  ? 'Shadow active'
                  : checkpoint.shadow_blocked_on !== null
                    ? 'Not a shadow model'
                    : 'Use as shadow'}
              </Button>
            </div>
          ))}
        </div>
      ) : null}

      {job ? (
        <div className="mb-4 rounded-lg border border-border p-3">
          <JobProgress job={job} />
        </div>
      ) : null}

      {performanceQuery.isPending ? (
        <div className="py-8 text-center text-xs text-text-muted">
          Scoring…
        </div>
      ) : performanceQuery.isError ? (
        <p className="text-xs text-red-400">
          {errMsg(performanceQuery.error)}
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[48rem] text-xs">
            <thead className="text-[10px] uppercase tracking-widest text-text-muted">
              <tr>
                {[
                  'policy',
                  'evaluation',
                  'reviewed',
                  'top-1',
                  'coverage',
                  'selected accuracy',
                  'occluded reject',
                ].map((heading, index) => (
                  <th
                    key={heading}
                    className={cn(
                      'px-2 py-1.5',
                      index < 2 ? 'text-left' : 'text-right',
                    )}
                  >
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(([name, metric, evaluation]) => (
                <tr key={name} className="border-t border-border">
                  <td className="px-2 py-1.5 font-mono text-text-primary">
                    {name}
                  </td>
                  <td className="px-2 py-1.5 text-text-muted">
                    {evaluation}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono">
                    {metric.reviewed}
                  </td>
                  <Metric value={metric.top1_accuracy} />
                  <Metric value={metric.auto_coverage} />
                  <Metric value={metric.selective_accuracy} />
                  <Metric value={metric.occluded_rejection_rate} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function Metric({ value }: { value: number | null }) {
  return (
    <td className="px-2 py-1.5 text-right font-mono">
      {value == null ? '—' : pct(value)}
    </td>
  );
}

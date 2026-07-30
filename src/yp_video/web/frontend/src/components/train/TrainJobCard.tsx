import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { JobProgress } from '@/components/job/JobProgress';
import { TrainDetail } from '@/components/train/TrainDetail';
import { isTerminal } from '@/lib/job';
import type { Job, TrainProgress } from '@/types/api';

interface TrainJobCardProps {
  job: Job | null;
  /** The job-params key this trainer streams its progress under. */
  progressKey: string;
  epochsFallback: number;
  onCancel?: () => void;
  mapLabel?: string;
  eventNoun?: string;
}

/** The "Training job" card every train page renders: live progress, the
 *  per-epoch detail strip, and a Cancel button while the job is running.
 *  Renders nothing until a job exists. */
export function TrainJobCard({ job, progressKey, epochsFallback, onCancel, mapLabel, eventNoun }: TrainJobCardProps) {
  if (!job) return null;
  return (
    <Card>
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <SectionLabel className="mb-0">Training job</SectionLabel>
        {onCancel && !isTerminal(job.status) ? (
          <Button size="sm" onClick={onCancel}>
            Cancel
          </Button>
        ) : null}
      </div>
      <JobProgress job={job} showLogs truncateMsg={false} />
      <TrainDetail
        progress={job.params?.[progressKey] as TrainProgress | undefined}
        epochsFallback={epochsFallback}
        mapLabel={mapLabel}
        eventNoun={eventNoun}
      />
    </Card>
  );
}

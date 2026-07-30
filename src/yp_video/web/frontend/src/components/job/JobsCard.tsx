import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { LiveJob } from '@/components/job/LiveJob';
import type { Job } from '@/types/api';

interface JobsCardProps {
  title: string;
  jobs: Job[];
  onUpdate: (job: Job) => void;
}

/** The "recent jobs" card the predict pages append: a LiveJob per row,
 *  hidden entirely while there is nothing to show. */
export function JobsCard({ title, jobs, onUpdate }: JobsCardProps) {
  if (!jobs.length) return null;
  return (
    <Card>
      <SectionLabel>{title}</SectionLabel>
      <div className="space-y-3">
        {jobs.map((job) => (
          <LiveJob key={job.id} job={job} onUpdate={onUpdate} />
        ))}
      </div>
    </Card>
  );
}

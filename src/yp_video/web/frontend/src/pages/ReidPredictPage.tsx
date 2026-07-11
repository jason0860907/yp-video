import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { API, ApiError, apiFetch } from '@/lib/api';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { LiveJob } from '@/components/job/LiveJob';
import { toast } from '@/components/feedback/toast';
import type { Job, ReidVideo } from '@/types/api';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

export function ReidPredictPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [overwrite, setOverwrite] = useState(false);
  const [stopVllm, setStopVllm] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);

  const videosQuery = useQuery({
    queryKey: ['reid-videos'],
    queryFn: () => apiFetch<ReidVideo[]>(API.reid.videos),
  });
  const videos = videosQuery.data ?? [];
  const extracted = videos.filter((v) => v.has_reid);

  const upsertJob = (job: Job) =>
    setJobs((prev) => (prev.some((j) => j.id === job.id) ? prev.map((j) => (j.id === job.id ? job : j)) : [job, ...prev]));

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    try {
      const job = await apiFetch<Job>(API.reid.start, {
        method: 'POST',
        body: { videos: names, overwrite, stop_vllm: stopVllm },
      });
      upsertJob(job);
      toast.success(`Started ReID Predict for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`ReID Predict start failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/reid-label')}>
              Open ReID Label
            </Button>
            <Button intent="primary" onClick={run}>
              Run ReID
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Extracted" value={extracted.length} tintClass="text-primary-light" />
        <StatTile label="Events" value={videos.reduce((s, v) => s + v.event_count, 0)} tintClass="text-text-muted" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        {/* Config */}
        <Card>
          <SectionLabel>Config</SectionLabel>
          <p className="mb-3 text-xs leading-relaxed text-text-muted">
            For every annotated action event: detect players on that frame (RF-DETR), pick the box the
            contact point belongs to, crop it and compute an OSNet appearance embedding. No tracking
            involved. Review and name the players on the ReID Label page afterwards.
          </p>
          <div className="space-y-2">
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Overwrite existing ReID results
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={stopVllm} onChange={(e) => setStopVllm(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Stop vLLM first
            </label>
          </div>
          <Button intent="primary" onClick={run} className="mt-4 w-full">
            Run ReID
          </Button>
        </Card>

        {/* Videos */}
        <Card>
          <VideoMultiSelectList
            videos={videos}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              { value: 'pending', label: 'No ReID', predicate: (v) => !v.has_reid },
              { value: 'all', label: 'All', predicate: () => true },
              { value: 'extracted', label: 'Extracted', predicate: (v) => v.has_reid },
            ]}
            renderMeta={(v) => (
              <>
                <span className="font-mono text-[11px] tabular-nums text-text-muted">{v.event_count}</span>
                {v.reid_counts && (
                  <Badge tone={v.reid_counts.miss > 0 ? 'warning' : 'success'}>
                    {v.reid_counts.ok + v.reid_counts.multi}/{v.event_count}
                  </Badge>
                )}
              </>
            )}
            emptyTitle="No annotated videos"
            emptySubtitle="Label some actions first — ReID runs on action events"
          />
        </Card>
      </div>

      {/* Jobs */}
      {jobs.length > 0 && (
        <Card>
          <SectionLabel>ReID Predict jobs</SectionLabel>
          <div className="space-y-3">
            {jobs.map((job) => (
              <LiveJob key={job.id} job={job} onUpdate={upsertJob} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

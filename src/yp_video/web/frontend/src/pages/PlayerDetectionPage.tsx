/** Player Detection: find everyone on screen when each action happened.
 *
 *  Perception, not judgement. RF-DETR Seg on every annotated
 *  action frame, keeping ALL the boxes — which of those people acted is
 *  Association Predict, and it re-decides among these boxes without ever
 *  opening the video again.
 *
 *  The sparse sibling of Rally Tracking: that one detects every frame of
 *  every rally, this one the ~300 frames an action happened on. Two
 *  perception stages, two upstreams, neither waiting on the other.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { API, ApiError, apiFetch } from '@/lib/api';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { PipelineChips, STAGE_HINT } from '@/components/video/PipelineChips';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { LiveJob } from '@/components/job/LiveJob';
import { toast } from '@/components/feedback/toast';
import type { ExtractionVideo, Job } from '@/types/api';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

const DETECT_JOB_TYPE = 'player_detection';

export function PlayerDetectionPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [overwrite, setOverwrite] = useState(false);
  const [stopVllm, setStopVllm] = useState(false);
  const [jobOverrides, setJobOverrides] = useState<Record<string, Job>>({});

  const jobsQuery = useQuery({
    queryKey: ['jobs-list'],
    queryFn: () => apiFetch<Job[]>(API.jobs.list),
  });
  const videosQuery = useQuery({
    queryKey: ['extraction-videos'],
    queryFn: () => apiFetch<ExtractionVideo[]>(API.extraction.videos),
  });
  const videos = videosQuery.data ?? [];
  const detected = videos.filter((v) => v.has_records);

  const upsertJob = (job: Job) => setJobOverrides((prev) => ({ ...prev, [job.id]: job }));
  const jobs = useMemo(() => {
    const merged = new Map<string, Job>();
    for (const job of jobsQuery.data ?? []) {
      if (job.type === DETECT_JOB_TYPE) merged.set(job.id, job);
    }
    for (const job of Object.values(jobOverrides)) merged.set(job.id, job);
    return [...merged.values()].sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0));
  }, [jobsQuery.data, jobOverrides]);

  // The action labels are the only prerequisite — they say which frames to
  // look at. Tracklets are the association stage's requirement, and gating
  // on them here would make two independent stages wait for each other.
  const chosen = videos.filter((v) => selected.has(v.name));
  const blocked = chosen.some((v) => !v.pipeline.has_action)
    ? STAGE_HINT.action
    : null;
  const undetected = chosen.filter((v) => !v.has_records).length;
  // Not a blocker, but worth saying before the run rather than after: these
  // will detect fine and then have nothing to associate with.
  const untracked = chosen.filter((v) => !v.pipeline.has_tracks).length;

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    try {
      const job = await apiFetch<Job>(API.extraction.detect, {
        method: 'POST',
        body: { videos: names, overwrite, stop_vllm: stopVllm },
      });
      upsertJob(job);
      toast.success(`Started Player Detection for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`Player Detection start failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/association-predict')}>
              Open Association Predict
            </Button>
            <Button intent="primary" onClick={run} disabled={Boolean(blocked)}>
              Run Player Detection
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Detected" value={detected.length} tintClass="text-primary-light" />
        <StatTile label="Events" value={videos.reduce((s, v) => s + v.event_count, 0)} tintClass="text-text-muted" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <Card>
          <SectionLabel>Config</SectionLabel>
          <p className="mb-3 text-xs leading-relaxed text-text-muted">
            For every annotated action event: detect every person on that frame with RF-DETR Seg
            and keep every box. Decides nothing — Association Predict picks who acted
            from this list, and never needs the video again.
          </p>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-text-secondary">Detector</span>
              <span className="font-mono text-text-muted">rf-detr-seg-medium</span>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Re-detect videos that already have detections
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={stopVllm} onChange={(e) => setStopVllm(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Stop vLLM first
            </label>
          </div>
          <Button
            intent="primary"
            onClick={run}
            disabled={Boolean(blocked)}
            className="mt-4 w-full"
            title={blocked ? `Cannot start: ${blocked}` : undefined}
          >
            Run Player Detection
            {!overwrite && undetected > 0 ? ` (${undetected} not detected)` : ''}
          </Button>
          {blocked && (
            <p className="mt-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-[11px] text-amber-400">
              {blocked}
            </p>
          )}
          {!blocked && untracked > 0 && (
            <p className="mt-2 rounded-lg border border-border-light bg-surface-50 px-3 py-1.5 text-[11px] text-text-muted">
              {untracked} selected video(s) have no tracklets yet. Detection does not need them —
              run it whenever. Association does, so run Rally Tracking before that.
            </p>
          )}
          <p className="mt-3 text-[11px] leading-relaxed text-text-muted">
            A re-detect refreshes the candidate list and keeps every pick already made — including
            human verdicts, which this stage has no opinion about.
          </p>
        </Card>

        <Card>
          <VideoMultiSelectList
            videos={videos}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              { value: 'pending', label: 'Not detected', predicate: (v) => !v.has_records },
              { value: 'all', label: 'All', predicate: () => true },
              { value: 'detected', label: 'Detected', predicate: (v) => v.has_records },
            ]}
            renderMeta={(v) => (
              <>
                <span className="font-mono text-[11px] tabular-nums text-text-muted">{v.event_count}</span>
                {v.detections ? <Badge tone="success">{v.detections} people</Badge> : null}
                {v.detector && <Badge tone="neutral">{v.detector}</Badge>}
                <PipelineChips pipeline={v.pipeline} />
              </>
            )}
            emptySubtitle="Label some actions first — detection runs on action events"
          />
        </Card>
      </div>

      {jobs.length > 0 && (
        <Card>
          <SectionLabel>Player Detection jobs</SectionLabel>
          <div className="space-y-2">
            {jobs.map((job) => (
              <LiveJob key={job.id} job={job} onUpdate={upsertJob} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

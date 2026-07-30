/** Rally Tracking: who is on court over time.
 *
 *  Its own page because it is its own stage. Tracking reads rally spans and
 *  nothing else — not the action labels, not extraction — so it runs in
 *  parallel with action labeling, and the button lived on whichever page
 *  happened to need tracklets rather than on the stage that produces them.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { API, apiFetch, errMsg } from '@/lib/api';
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
import { useTypedJobs } from '@/lib/useTypedJobs';
import type { ExtractionVideo, Job } from '@/types/api';


const TRACKING_JOB_TYPE = 'player_tracking';

export function TrackingPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [overwrite, setOverwrite] = useState(false);
  const [stopVllm, setStopVllm] = useState(false);
  const [stride, setStride] = useState(1);
  const { jobs, upsertJob } = useTypedJobs([TRACKING_JOB_TYPE]);

  // The extraction listing is the one that carries every cut with its
  // pipeline state; tracking only reads the rally half of it.
  const videosQuery = useQuery({
    queryKey: ['extraction-videos'],
    queryFn: () => apiFetch<ExtractionVideo[]>(API.extraction.videos),
  });
  const videos = videosQuery.data ?? [];
  const tracked = videos.filter((v) => v.pipeline.has_tracks);

  const chosen = videos.filter((v) => selected.has(v.name));
  const blocked = chosen.some((v) => v.pipeline.rally_sources.length === 0)
    ? STAGE_HINT.rallies
    : null;
  // Videos in the selection that would actually gain something.
  const missing = chosen.filter((v) => !v.pipeline.has_tracks).length;

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    try {
      const job = await apiFetch<Job>(API.tracklets.run, {
        method: 'POST',
        body: { videos: names, overwrite, stop_vllm: stopVllm, stride },
      });
      upsertJob(job);
      toast.success(`Started Rally Tracking for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`Rally Tracking start failed: ${errMsg(e)}`);
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
              Run Rally Tracking
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Tracked" value={tracked.length} tintClass="text-primary-light" />
        <StatTile label="Untracked in selection" value={missing} tintClass="text-text-muted" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <Card>
          <SectionLabel>Config</SectionLabel>
          <p className="mb-3 text-xs leading-relaxed text-text-muted">
            Dense RF-DETR Seg detection over every frame of every rally span, linked into tracklets by
            ByteTrack — one tracker per rally, because between rallies players reshuffle. Needs rally
            spans only, so it can run while actions are still being labeled.
          </p>
          <div className="space-y-2">
            <label className="block text-xs text-text-secondary">
              <span className="mb-1 block">
                Stride <span className="text-text-muted">— detect every Nth rally frame</span>
              </span>
              <input
                type="number"
                min={1}
                max={10}
                value={stride}
                onChange={(e) => setStride(Math.min(10, Math.max(1, Number(e.target.value) || 1)))}
                className="w-full rounded-lg border border-border-light bg-surface-50 px-3 py-1.5 text-xs text-text-primary focus:border-primary/50 focus:outline-none"
              />
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Overwrite existing tracklets
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
            Run Rally Tracking
          </Button>
          {blocked && (
            <p className="mt-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-[11px] text-amber-400">
              {blocked}
            </p>
          )}
          <p className="mt-3 text-[11px] leading-relaxed text-text-muted">
            Re-tracking renumbers every <code>track_id</code> — a tracklet is keyed{' '}
            <code>{'{rally_id}:{track_id}'}</code>, so labels attached to tracklets have to be
            re-anchored afterwards. A <Badge tone="warning">stale</Badge> chip means the rallies
            moved since these tracklets were cut.
          </p>
        </Card>

        <Card>
          <VideoMultiSelectList
            videos={videos}
            query={videosQuery}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              { value: 'pending', label: 'Untracked', predicate: (v) => !v.pipeline.has_tracks },
              { value: 'all', label: 'All', predicate: () => true },
              { value: 'tracked', label: 'Tracked', predicate: (v) => v.pipeline.has_tracks },
            ]}
            renderMeta={(v) => <PipelineChips pipeline={v.pipeline} />}
            emptySubtitle="Label some rallies first — tracking runs on rally spans"
          />
        </Card>
      </div>

      {jobs.length > 0 && (
        <Card>
          <SectionLabel>Rally Tracking jobs</SectionLabel>
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

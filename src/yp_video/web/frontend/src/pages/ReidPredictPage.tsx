/** ReID Predict: turn reviewed crops into embedding vectors.
 *
 *  All this stage does is compute the vectors the ReID Label board groups on
 *  — it never opens the video. Run it once the actors are reviewed on
 *  Association Label: an embedding answers "who is this person" about a crop,
 *  so a crop that review is about to re-cut is a vector thrown away.
 *
 *  This page used to run extraction and tracking too, which is how "ReID"
 *  became the name for three stages that merely feed it.
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
import type { Job, ReidOptions, ReidVideo } from '@/types/api';


// The one job this page runs — rehydrated from the server list so navigating
// away and back doesn't lose live progress.
const EMBED_JOB_TYPE = 'player_embed';

export function ReidPredictPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [overwrite, setOverwrite] = useState(false);
  const [stopVllm, setStopVllm] = useState(false);
  // '' = the fixed official checkpoint; else an explicitly selected candidate.
  const [checkpoint, setCheckpoint] = useState('');
  const { jobs, upsertJob } = useTypedJobs([EMBED_JOB_TYPE]);

  const videosQuery = useQuery({
    queryKey: ['reid-videos'],
    queryFn: () => apiFetch<ReidVideo[]>(API.reid.videos),
  });
  const optionsQuery = useQuery({
    queryKey: ['reid-options'],
    queryFn: () => apiFetch<ReidOptions>(API.reid.options),
    staleTime: Infinity, // static per server run
  });
  const registeredEmbedders = (optionsQuery.data?.embedders ?? []).map((e) => e.name);
  const checkpoints = optionsQuery.data?.checkpoints ?? [];
  const videos = videosQuery.data ?? [];
  const embedded = videos.filter((v) => v.embedded_models.length > 0);

  // Embedding reads the saved crops, so the only prerequisite is that they
  // exist — a disabled button with a reason beats a 400 after the click.
  const chosen = videos.filter((v) => selected.has(v.name));
  const blocked = chosen.some((v) => !v.pipeline.has_records) ? STAGE_HINT.records : null;
  const notEmbedded = chosen.filter((v) => v.embedded_models.length === 0).length;

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    try {
      const job = await apiFetch<Job>(API.reid.embed, {
        method: 'POST',
        body: { videos: names, overwrite, stop_vllm: stopVllm, checkpoint: checkpoint || null },
      });
      upsertJob(job);
      toast.success(`Started embedding for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`Embedding start failed: ${errMsg(e)}`);
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
            <Button intent="primary" onClick={run} disabled={Boolean(blocked)}>
              Run Embedding
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Embedded" value={embedded.length} tintClass="text-primary-light" />
        <StatTile label="Not embedded in selection" value={notEmbedded} tintClass="text-text-muted" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <Card>
          <SectionLabel>Config</SectionLabel>
          <p className="mb-3 text-xs leading-relaxed text-text-muted">
            Every saved crop → one appearance vector per registered embedder. The video file is never
            opened, so this is also how a newly registered model covers already-extracted videos.
          </p>
          <p className="mb-3 rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-[11px] leading-relaxed text-text-muted">
            Review the actors on <strong>Association Label</strong> first. A fix there re-cuts the
            crop, and the vector of the crop it replaced is discarded — embedding after the review
            costs one pass instead of one per fix.
          </p>
          <div className="space-y-2">
            {checkpoints.length > 0 && (
              <label className="block text-xs text-text-secondary">
                <span className="mb-1 block">
                  Checkpoint <span className="text-text-muted">— clip-reident weights to embed with</span>
                </span>
                <select
                  value={checkpoint}
                  onChange={(e) => setCheckpoint(e.target.value)}
                  className="w-full cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-1.5 text-xs text-text-primary focus:border-primary/50 focus:outline-none"
                >
                  <option value="">Default (official) — {checkpoints.find((c) => c.active)?.run_name ?? 'none'}</option>
                  {checkpoints.map((c) => (
                    <option key={c.ref} value={c.ref}>
                      {c.run_name}
                      {c.active ? ' (active)' : ''}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Overwrite — recompute every model, not just the missing ones
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
            Run Embedding
          </Button>
          {blocked && (
            <p className="mt-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-[11px] text-amber-400">
              {blocked}
            </p>
          )}
        </Card>

        <Card>
          <VideoMultiSelectList
            videos={videos}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              { value: 'pending', label: 'Not embedded', predicate: (v) => v.embedded_models.length === 0 },
              { value: 'all', label: 'All', predicate: () => true },
              { value: 'embedded', label: 'Embedded', predicate: (v) => v.embedded_models.length > 0 },
            ]}
            renderMeta={(v) => {
              const missing = v.pipeline.has_records
                ? registeredEmbedders.filter((m) => !v.embedded_models.includes(m))
                : [];
              const stale = v.stale_embedding_models ?? [];
              return (
                <>
                  <span className="font-mono text-[11px] tabular-nums text-text-muted">{v.event_count}</span>
                  {v.embedded_models.length > 0 && (
                    <Badge tone="success">{v.embedded_models.length} models</Badge>
                  )}
                  {missing.length > 0 && (
                    <span title="Registered embedders with no matrix for this video — run Embedding">
                      <Badge tone="warning">missing: {missing.join(', ')}</Badge>
                    </span>
                  )}
                  {stale.length > 0 && (
                    <span title="Refreshing after an actor fix">
                      <Badge tone="warning">stale: {stale.join(', ')}</Badge>
                    </span>
                  )}
                  <PipelineChips pipeline={v.pipeline} />
                </>
              );
            }}
            emptySubtitle="Label some actions first — embedding runs on extracted crops"
          />
        </Card>
      </div>

      {jobs.length > 0 && (
        <Card>
          <SectionLabel>Embedding jobs</SectionLabel>
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

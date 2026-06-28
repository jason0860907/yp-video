import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { LiveJob } from '@/components/job/LiveJob';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { ActionVideo, Job, SpotInfo } from '@/types/api';

interface PredSettings {
  checkpoint: string;
  min_score: number;
  batch_size: number;
  clip_len: number;
  decoder: 'opencv' | 'nvdec';
  decode_producers: number;
  decoder_threads: number;
  prefetch_factor: number;
  decode_chunk_frames: number;
  overwrite: boolean;
  stop_vllm: boolean;
}
const DEFAULTS: PredSettings = {
  checkpoint: '',
  min_score: 0.15,
  batch_size: 64,
  clip_len: 64,
  decoder: 'nvdec',
  decode_producers: 2,
  decoder_threads: 1,
  prefetch_factor: 2,
  decode_chunk_frames: 256,
  overwrite: false,
  stop_vllm: false,
};

const NUM_FIELDS: Array<{ key: keyof PredSettings; label: string; min: number; max?: number; step: number }> = [
  { key: 'min_score', label: 'Min score', min: 0, max: 1, step: 0.05 },
  { key: 'batch_size', label: 'Batch', min: 1, max: 128, step: 1 },
  { key: 'clip_len', label: 'Clip len', min: 8, max: 256, step: 8 },
  { key: 'decode_producers', label: 'Producers', min: 1, max: 8, step: 1 },
  { key: 'decoder_threads', label: 'Threads', min: 1, max: 8, step: 1 },
  { key: 'prefetch_factor', label: 'Prefetch', min: 1, max: 8, step: 1 },
  { key: 'decode_chunk_frames', label: 'Chunk', min: 1, max: 512, step: 16 },
];

type StatusFilter = 'unlabeled' | 'all' | 'labeled';
type KindFilter = 'all' | 'broadcast' | 'sideline';

const fieldCls =
  'w-full rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';
const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const hasLabels = (v: ActionVideo) =>
  Boolean(v.has_action_annotation || v.has_action_final_annotation || v.has_action_pre_annotation);

export function ActionPredictPage() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('unlabeled');
  const [settings, setSettings] = useState<PredSettings>(DEFAULTS);
  const [jobs, setJobs] = useState<Job[]>([]);

  const videosQuery = useQuery({
    queryKey: ['action-videos'],
    queryFn: () => apiFetch<ActionVideo[]>(API.actionAnnotate.videos),
  });
  const spotQuery = useQuery({
    queryKey: ['spot-info'],
    queryFn: () => apiFetch<SpotInfo>(API.actionAnnotate.spot),
  });

  const videos = videosQuery.data ?? [];
  const spot = spotQuery.data;

  // Seed checkpoint from the server default once available.
  useEffect(() => {
    if (spot?.default_checkpoint && !settings.checkpoint) {
      setSettings((s) => ({ ...s, checkpoint: spot.default_checkpoint! }));
    }
  }, [spot?.default_checkpoint, settings.checkpoint]);

  const visible = videos.filter((v) => {
    if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
    if (statusFilter === 'unlabeled' && hasLabels(v)) return false;
    if (statusFilter === 'labeled' && !hasLabels(v)) return false;
    return true;
  });

  const labeledCount = videos.filter(hasLabels).length;
  const runningCount = jobs.filter((j) => j.status === 'running').length;
  const checkpoints = spot?.checkpoints ?? [];
  const spotReady = Boolean(spot?.available && checkpoints.length);

  const toggle = (name: string, on: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (on) next.add(name);
      else next.delete(name);
      return next;
    });
  const allVisibleSelected = visible.length > 0 && visible.every((v) => selected.has(v.name));
  const toggleVisible = (on: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      visible.forEach((v) => (on ? next.add(v.name) : next.delete(v.name)));
      return next;
    });

  const upsertJob = (job: Job) => setJobs((prev) => (prev.some((j) => j.id === job.id) ? prev.map((j) => (j.id === job.id ? job : j)) : [job, ...prev]));

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    const existing = names.map((n) => videos.find((v) => v.name === n)).filter((v): v is ActionVideo => Boolean(v)).filter(hasLabels);
    if (existing.length && !settings.overwrite) {
      toast.warning(`${existing.length} selected video(s) already have action labels`);
      return;
    }
    if (existing.length && settings.overwrite) {
      const ok = await confirm({
        title: 'Overwrite action labels?',
        body: `This replaces the active action labels for ${existing.length} video(s) with new pre-labels.`,
        confirmText: 'Overwrite',
        variant: 'danger',
      });
      if (!ok) return;
    }
    try {
      const job = await apiFetch<Job>(API.actionAnnotate.prelabelBatch, {
        method: 'POST',
        body: {
          videos: names,
          checkpoint: settings.checkpoint,
          min_score: settings.min_score,
          batch_size: settings.batch_size,
          clip_len: settings.clip_len,
          num_workers: settings.decode_producers,
          decoder: settings.decoder,
          decode_producers: settings.decode_producers,
          decoder_threads: settings.decoder_threads,
          prefetch_factor: settings.prefetch_factor,
          decode_chunk_frames: settings.decode_chunk_frames,
          use_amp: true,
          overwrite: settings.overwrite,
          stop_vllm: settings.stop_vllm,
        },
      });
      upsertJob(job);
      toast.success(`Started SPOT batch for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`SPOT start failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        eyebrow="PIPELINE · ACTION · SPOT"
        title="Action Predict"
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/action-annotate')}>
              Open Label
            </Button>
            <Button intent="primary" onClick={run} disabled={!spotReady}>
              Run SPOT
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-accent" />
        <StatTile label="Labeled" value={labeledCount} tintClass="text-emerald-400" />
        <StatTile label="Running" value={runningCount} tintClass={runningCount ? 'text-primary-light' : 'text-text-muted'} />
      </div>

      {!spotReady && (
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.06] px-4 py-3 text-sm text-amber-300">
          SPOT unavailable: {spot?.error || (spot?.available ? 'no checkpoint found' : '~/yp-spot not ready')}
        </div>
      )}

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1fr_1.6fr]">
        {/* Config */}
        <Card>
          <SectionLabel>Config</SectionLabel>
          <label className="mb-1.5 block text-[10.5px] uppercase tracking-wide text-text-muted">Checkpoint</label>
          <select
            value={settings.checkpoint}
            onChange={(e) => setSettings((s) => ({ ...s, checkpoint: e.target.value }))}
            className={cn(fieldCls, 'mb-3 cursor-pointer appearance-none')}
          >
            {checkpoints.length === 0 && <option value="">No checkpoint</option>}
            {checkpoints.map((c) => (
              <option key={c.path} value={c.path}>
                {c.name} · {c.is_best ? 'best' : `epoch ${c.epoch}`}
              </option>
            ))}
          </select>

          <div className="grid grid-cols-2 gap-2.5">
            {NUM_FIELDS.map((f) => (
              <div key={f.key}>
                <label className="mb-1 block text-[10px] uppercase tracking-wide text-text-muted">{f.label}</label>
                <input
                  type="number"
                  value={settings[f.key] as number}
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  onChange={(e) => setSettings((s) => ({ ...s, [f.key]: Number(e.target.value) }))}
                  className={cn(fieldCls, 'font-mono tabular-nums')}
                />
              </div>
            ))}
            <div>
              <label className="mb-1 block text-[10px] uppercase tracking-wide text-text-muted">Decoder</label>
              <select
                value={settings.decoder}
                onChange={(e) => setSettings((s) => ({ ...s, decoder: e.target.value as PredSettings['decoder'] }))}
                className={cn(fieldCls, 'cursor-pointer appearance-none')}
              >
                <option value="nvdec">NVDEC (GPU)</option>
                <option value="opencv">OpenCV</option>
              </select>
            </div>
          </div>

          <div className="mt-3 space-y-2">
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={settings.overwrite}
                onChange={(e) => setSettings((s) => ({ ...s, overwrite: e.target.checked }))}
                className="h-3.5 w-3.5 accent-primary"
              />
              Overwrite existing action labels
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={settings.stop_vllm}
                onChange={(e) => setSettings((s) => ({ ...s, stop_vllm: e.target.checked }))}
                className="h-3.5 w-3.5 accent-primary"
              />
              Stop vLLM first
            </label>
          </div>

          <Button intent="primary" onClick={run} disabled={!spotReady} className="mt-4 w-full">
            Run SPOT
          </Button>
        </Card>

        {/* Videos */}
        <Card>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <SectionLabel className="mb-0">Videos</SectionLabel>
            <div className="flex items-center gap-2">
              <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as KindFilter)} className={cn(fieldCls, 'w-auto cursor-pointer appearance-none py-1 text-xs')}>
                <option value="all">All kinds</option>
                <option value="broadcast">Broadcast</option>
                <option value="sideline">Sideline</option>
              </select>
              <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as StatusFilter)} className={cn(fieldCls, 'w-auto cursor-pointer appearance-none py-1 text-xs')}>
                <option value="unlabeled">Unlabeled</option>
                <option value="all">All</option>
                <option value="labeled">Labeled</option>
              </select>
            </div>
          </div>
          <div className="mb-2 flex items-center justify-between text-xs text-text-muted">
            <label className="inline-flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={allVisibleSelected} onChange={(e) => toggleVisible(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Select visible
            </label>
            <span className="font-mono tabular-nums">
              {selected.size} selected / {visible.length} shown
            </span>
          </div>

          <div className="max-h-[56vh] space-y-1 overflow-y-auto pr-1">
            {visible.length === 0 ? (
              <EmptyState
                icon={
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3" />
                  </svg>
                }
                title="No videos"
              />
            ) : (
              visible.map((v) => (
                <label
                  key={v.name}
                  className="flex cursor-pointer items-center gap-3 rounded-lg border border-border bg-surface-50 px-3 py-2 transition-colors hover:border-border-light"
                >
                  <input type="checkbox" checked={selected.has(v.name)} onChange={(e) => toggle(v.name, e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
                  <span className={cn('h-2 w-2 flex-shrink-0 rounded-full', v.kind === 'broadcast' ? 'bg-primary-light' : 'bg-accent-light')} />
                  <span className="min-w-0 flex-1 truncate text-sm text-text-primary">{v.name}</span>
                  <span className={cn('font-mono text-[11px] tabular-nums', hasLabels(v) ? 'text-emerald-300' : 'text-text-muted')}>
                    {v.event_count || 0}
                  </span>
                </label>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* Jobs */}
      {jobs.length > 0 && (
        <Card>
          <SectionLabel>SPOT jobs</SectionLabel>
          <div className="space-y-3">
            {jobs.map((job) => (
              <LiveJob
                key={job.id}
                job={job}
                onUpdate={upsertJob}
                onSettled={(j) => {
                  if (j.status === 'completed') void qc.invalidateQueries({ queryKey: ['action-videos'] });
                }}
              />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

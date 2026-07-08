import { useEffect, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { isTerminal } from '@/lib/job';
import { useSSE } from '@/lib/useSSE';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { JobProgress } from '@/components/job/JobProgress';
import { ActionPerfCharts } from '@/components/train/ActionPerfCharts';
import { toast } from '@/components/feedback/toast';
import type { ActionPerfData, Job, RallyMapBreakdown, RallyTrainProgress, RallyTrainStatus } from '@/types/api';

interface Form {
  extract_fps: number;
  video_limit: number;
  camera_view: 'all' | 'broadcast' | 'sideline';
  init_checkpoint: string;
  resume_run: string; // '' = fresh run; else the exp/ save_dir to --resume
  feature_arch: string;
  temporal_arch: string;
  num_epochs: number;
  batch_size: number;
  clip_len: number;
  num_workers: number;
  gpu: number;
  learning_rate: number;
  warm_up_epochs: number;
  criterion: string;
  start_val_epoch: number;
  epoch_num_frames: number | '';
  val_ratio: number;
  split_seed: number;
  stop_vllm: boolean;
}

const BASE_FORM: Form = {
  extract_fps: 2,
  video_limit: 100,
  camera_view: 'all',
  init_checkpoint: '',
  resume_run: '',
  feature_arch: 'rny008_gsm',
  temporal_arch: 'gru',
  num_epochs: 30,
  batch_size: 8,
  clip_len: 64,
  num_workers: 4,
  gpu: 0,
  learning_rate: 0.0003,
  warm_up_epochs: 2,
  criterion: 'map',
  start_val_epoch: 0,
  epoch_num_frames: '',
  val_ratio: 0.2,
  split_seed: 42,
  stop_vllm: false,
};

const SELECTS = {
  feature_arch: ['rny008_gsm', 'rny002_gsm', 'convnextt_gsm', 'rn18_gsm'],
  temporal_arch: ['gru', 'deeper_gru', 'mingru'],
  criterion: ['map', 'loss'],
} as const;

const NUM_FIELDS: Array<{ key: keyof Form; label: string; min?: number; max?: number; step?: number }> = [
  { key: 'video_limit', label: 'Video limit', min: 0, max: 2000 },
  { key: 'num_epochs', label: 'Epochs', min: 1, max: 1000 },
  { key: 'batch_size', label: 'Batch', min: 1, max: 64 },
  { key: 'clip_len', label: 'Clip len', min: 8, max: 256 },
  { key: 'num_workers', label: 'Workers', min: 0, max: 32 },
  { key: 'gpu', label: 'GPU', min: 0, max: 7 },
  { key: 'learning_rate', label: 'LR', min: 0, step: 0.0001 },
  { key: 'warm_up_epochs', label: 'Warmup', min: 0, max: 100 },
  { key: 'start_val_epoch', label: 'Start val', min: 0, max: 1000 },
];

const fieldCls =
  'w-full rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';
const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const fmtMetric = (v: unknown) => (Number.isFinite(Number(v)) ? Number(v).toFixed(4) : '');

export function SpotTrainPage() {
  const [form, setForm] = useState<Form>(BASE_FORM);
  const [job, setJob] = useState<Job | null>(null);

  const statusQuery = useQuery({
    queryKey: ['spot-train-status'],
    queryFn: () => apiFetch<RallyTrainStatus>(API.spotTrain.status),
    refetchInterval: job && !isTerminal(job.status) ? false : 20_000,
  });
  const status = statusQuery.data;

  const perfQuery = useQuery({
    queryKey: ['spot-train-performance'],
    queryFn: () => apiFetch<ActionPerfData>(API.spotTrain.performance),
    refetchInterval: job && !isTerminal(job.status) ? 30_000 : false,
  });
  const perf = perfQuery.data;

  // Adopt any active job on first load.
  useEffect(() => {
    const active = status?.active_job;
    if (active && !job) setJob(active);
  }, [status?.active_job, job]);

  useSSE<Job>(job && !isTerminal(job.status) ? API.jobs.eventsSSE(job.id) : null, (data) => {
    setJob(data);
    if (isTerminal(data.status)) {
      if (data.status === 'completed') toast.success('SPOT rally training complete');
      if (data.status === 'failed') toast.error(`SPOT rally training failed: ${data.error || data.message}`);
    }
  });

  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));

  const ann = status?.rally_annotations;
  const usable = Math.max(0, Number(ann?.with_local_video) || 0);
  const trainingVideos = form.video_limit > 0 ? Math.min(form.video_limit, usable) : usable;
  const ready = usable > 0;
  const running = !!job && (job.status === 'running' || job.status === 'pending');
  const canStart = !running && ready && Boolean(status?.spot_available);
  const initCheckpoints = status?.init_checkpoints ?? [];
  const resumableRuns = status?.resumable_runs ?? [];
  const isResuming = form.resume_run !== '';
  // Rough JPEG footprint of the frame cache this run would need (~15 KB/frame).
  const estCacheGb = ((Number(ann?.total_hours) || 0) * (trainingVideos / Math.max(1, usable)) * 3600 * form.extract_fps * 15) / 1e6;

  const start = async () => {
    try {
      const body = {
        extract_fps: form.extract_fps,
        video_limit: form.video_limit,
        camera_view: form.camera_view,
        init_checkpoint: form.resume_run !== '' ? null : form.init_checkpoint.trim() || null,
        resume: form.resume_run !== '',
        save_dir: form.resume_run || null,
        gpu: form.gpu,
        feature_arch: form.feature_arch,
        temporal_arch: form.temporal_arch,
        clip_len: form.clip_len,
        batch_size: form.batch_size,
        num_epochs: form.num_epochs,
        warm_up_epochs: form.warm_up_epochs,
        learning_rate: form.learning_rate,
        num_workers: form.num_workers,
        criterion: form.criterion,
        start_val_epoch: form.start_val_epoch,
        epoch_num_frames: form.epoch_num_frames === '' ? null : form.epoch_num_frames,
        val_ratio: form.val_ratio,
        split_seed: form.split_seed,
        stop_vllm: form.stop_vllm,
      };
      const started = await apiFetch<Job>(API.spotTrain.start, { method: 'POST', body });
      setJob(started);
      toast.success('SPOT rally training started');
    } catch (e) {
      toast.error(`SPOT rally training failed to start: ${errMsg(e)}`);
    }
  };

  const cancel = async () => {
    if (!job?.id) return;
    try {
      await apiFetch(API.jobs.cancel(job.id), { method: 'POST' });
      toast.warning('SPOT rally training cancelled');
    } catch (e) {
      toast.error(`Cancel failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Annotated videos" value={`${usable}/${ann?.videos ?? 0}`} tintClass="text-primary-light" />
        <StatTile label="Rallies" value={(ann?.rallies ?? 0).toLocaleString()} tintClass="text-primary-light" />
        <StatTile label="Rally hours" value={(Number(ann?.rally_hours) || 0).toFixed(1)} tintClass="text-primary-light" />
        <StatTile label="Checkpoints" value={status?.rally_checkpoints?.runs ?? 0} tintClass="text-primary-light" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1.6fr_1fr]">
        {/* Training config */}
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
            <Field label="Extract fps">
              <select
                value={form.extract_fps}
                onChange={(e) => set('extract_fps', Number(e.target.value))}
                className={cn(fieldCls, 'cursor-pointer appearance-none')}
              >
                {[1, 2, 5].map((fps) => (
                  <option key={fps} value={fps}>
                    {fps} fps
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Init checkpoint" className="col-span-2">
              <select
                value={form.init_checkpoint}
                onChange={(e) => set('init_checkpoint', e.target.value)}
                title={isResuming ? 'Ignored while resuming (weights load from the run checkpoint)' : form.init_checkpoint}
                disabled={isResuming}
                className={cn(fieldCls, 'cursor-pointer appearance-none', isResuming && 'opacity-50')}
              >
                <option value="">— From scratch —</option>
                {initCheckpoints.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Resume from run" className="col-span-3">
              <select
                value={form.resume_run}
                onChange={(e) => set('resume_run', e.target.value)}
                title={form.resume_run}
                className={cn(fieldCls, 'cursor-pointer appearance-none')}
              >
                <option value="">— New run (train from scratch / init checkpoint) —</option>
                {resumableRuns.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Feature">
              <SelectArch value={form.feature_arch} options={SELECTS.feature_arch} onChange={(v) => set('feature_arch', v)} />
            </Field>
            <Field label="Temporal">
              <SelectArch value={form.temporal_arch} options={SELECTS.temporal_arch} onChange={(v) => set('temporal_arch', v)} />
            </Field>
            <Field label="Camera view">
              <select value={form.camera_view} onChange={(e) => set('camera_view', e.target.value as Form['camera_view'])} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
                <option value="all">All Views</option>
                <option value="broadcast">Broadcast</option>
                <option value="sideline">Sideline</option>
              </select>
            </Field>

            {NUM_FIELDS.map((f) => (
              <Field key={f.key} label={f.label}>
                <input
                  type="number"
                  value={form[f.key] as number}
                  min={f.min}
                  max={f.max}
                  step={f.step ?? 1}
                  onChange={(e) => set(f.key, Number(e.target.value) as Form[typeof f.key])}
                  className={cn(fieldCls, 'font-mono tabular-nums')}
                />
              </Field>
            ))}
            <Field label="Criterion">
              <SelectArch value={form.criterion} options={SELECTS.criterion} onChange={(v) => set('criterion', v)} />
            </Field>
            <Field label="Epoch frames">
              <input
                type="number"
                value={form.epoch_num_frames}
                min={1}
                placeholder="optional"
                onChange={(e) => set('epoch_num_frames', e.target.value === '' ? '' : Number(e.target.value))}
                className={cn(fieldCls, 'font-mono tabular-nums')}
              />
            </Field>
            <Field label="Val ratio">
              <input type="number" value={form.val_ratio} min={0.01} max={0.9} step={0.01} onChange={(e) => set('val_ratio', Number(e.target.value))} className={cn(fieldCls, 'font-mono tabular-nums')} />
            </Field>
            <Field label="Split seed">
              <input type="number" value={form.split_seed} onChange={(e) => set('split_seed', Number(e.target.value))} className={cn(fieldCls, 'font-mono tabular-nums')} />
            </Field>
          </div>

          <p className="mt-2 text-xs text-text-secondary">
            Trains on {trainingVideos} video(s); frames are extracted once at {form.extract_fps} fps (~{estCacheGb.toFixed(0)} GB cache) and reused. Video limit 0 = all annotated videos.
          </p>

          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-text-secondary">
            <label className="inline-flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={form.stop_vllm} onChange={(e) => set('stop_vllm', e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Stop vLLM
            </label>
          </div>

          <div className="mt-4 flex items-center gap-2">
            <Button intent="primary" onClick={start} disabled={!canStart} className="flex-1">
              {running ? 'Training…' : 'Start Training'}
            </Button>
            {running && <Button onClick={cancel}>Cancel</Button>}
          </div>
        </Card>

        {/* Dataset summary */}
        <Card>
          <SectionLabel>Rally Labels</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {[
              ['Labels', `${usable} vid / ${(ann?.rallies ?? 0).toLocaleString()} rallies`],
              ['Coverage', `${(Number(ann?.rally_hours) || 0).toFixed(1)}h rally / ${(Number(ann?.total_hours) || 0).toFixed(1)}h video`],
              ['Missing', `${ann?.missing_videos ?? 0} annotation(s) without local video`],
              ['View', form.camera_view === 'all' ? 'all views' : form.camera_view],
              ['Label dir', ann?.label_dir || '—'],
              ['Ckpt dir', status?.rally_checkpoints?.dir ? `${status.rally_checkpoints.dir}/<auto run>` : '—'],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-16 flex-shrink-0 text-text-muted">{label}</span>
                <span className="min-w-0 flex-1 truncate font-mono tabular-nums text-text-secondary" title={String(value)}>
                  {value}
                </span>
              </div>
            ))}
          </div>

          {(status?.frame_caches?.length ?? 0) > 0 && (
            <div className="mt-3 space-y-1 text-[11.5px]">
              <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted">Frame caches</div>
              {status!.frame_caches!.map((c) => (
                <div key={c.fps} className="flex items-center gap-3">
                  <span className="w-16 flex-shrink-0 text-text-muted">{c.fps} fps</span>
                  <span className="font-mono tabular-nums text-text-secondary">{c.videos} video(s) cached</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Training job */}
      {job && (
        <Card>
          <SectionLabel>Training job</SectionLabel>
          <JobProgress job={job} showLogs truncateMsg={false} />
          <TrainDetail progress={job.params?.rally_train_progress as RallyTrainProgress | undefined} epochsFallback={form.num_epochs} />
        </Card>
      )}

      {/* Per-epoch curve + per-video mAP for the latest (or running) run */}
      {perf && perf.entries.length > 0 && <ActionPerfCharts data={perf} />}
    </div>
  );
}

function Field({ label, className, children }: { label: string; className?: string; children: ReactNode }) {
  return (
    <label className={cn('block min-w-0 space-y-1', className)}>
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

function SelectArch({ value, options, onChange }: { value: string; options: readonly string[]; onChange: (v: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function TrainDetail({ progress: p, epochsFallback }: { progress?: RallyTrainProgress; epochsFallback: number }) {
  if (!p) return null;
  const phaseProgress = Number.isFinite(Number(p.phase_progress)) ? `${Math.round(Number(p.phase_progress) * 100)}%` : '';
  const step =
    Number.isFinite(Number(p.step)) && Number.isFinite(Number(p.total)) ? `${p.step}/${p.total}${phaseProgress ? ` (${phaseProgress})` : ''}` : '';
  const latestMap = Number.isFinite(Number(p.latest_val_map)) ? `${(Number(p.latest_val_map) * 100).toFixed(2)}%` : '';
  const bestVal = Number.isFinite(Number(p.best_value))
    ? Number(p.best_value) <= 1
      ? (Number(p.best_value) * 100).toFixed(2) + '%'
      : Number(p.best_value).toFixed(4)
    : '';
  const best = bestVal ? `${bestVal}${p.best_epoch != null ? ` · Epoch ${Number(p.best_epoch) + 1}` : ''}` : '';
  const rows: Array<[string, string]> = [
    ['Epoch', `${p.epoch_display || 1}/${p.epochs || epochsFallback || '?'}`],
    ['Phase', p.phase_label || p.phase || ''],
    ['Step', step],
    ['Cur loss', fmtMetric(p.current_loss)],
    ['Last train', fmtMetric(p.latest_train_loss)],
    ['Last val', fmtMetric(p.latest_val_loss)],
    ['Seg mAP', latestMap],
    ['Best', best],
  ];
  const visible = rows.filter(([, v]) => v !== '' && v != null);
  if (!visible.length) return null;
  return (
    <>
      <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
        {visible.map(([label, value]) => (
          <div key={label} className="min-w-0 rounded-lg border border-border bg-surface-100 px-2.5 py-2">
            <div className="text-[9px] uppercase tracking-wider text-text-muted">{label}</div>
            <div className="mt-0.5 truncate font-mono text-[11px] tabular-nums text-text-secondary" title={value}>
              {value}
            </div>
          </div>
        ))}
      </div>
      {(p.latest_val_breakdown || p.best_breakdown) && (
        <details className="mt-2">
          <summary className="cursor-pointer text-[10px] text-text-muted hover:text-text-primary">Segment mAP breakdown</summary>
          {p.latest_val_breakdown && <SegmentBreakdownTable title={`Latest — Epoch ${p.epoch_display ?? 1}`} bd={p.latest_val_breakdown} />}
          {p.best_breakdown && <SegmentBreakdownTable title={`Best${p.best_epoch != null ? ` — Epoch ${p.best_epoch + 1}` : ''}`} bd={p.best_breakdown} />}
        </details>
      )}
    </>
  );
}

function SegmentBreakdownTable({ title, bd }: { title: string; bd: RallyMapBreakdown }) {
  const pct = (v: number | undefined) => (Number.isFinite(v) ? ((v as number) * 100).toFixed(1) : '—');
  const numCell = 'py-0.5 pl-5 text-right';
  return (
    <div className="mt-3">
      <div className="text-xs font-semibold text-text-primary">{title}</div>
      <div className="mt-1.5 grid grid-cols-1 items-start gap-3 xl:grid-cols-[auto_minmax(0,1fr)]">
        {/* AP per class per temporal-IoU threshold */}
        <div className="rounded-lg border border-border bg-surface-100 px-3 py-2.5">
          <div className="text-[9px] uppercase tracking-wider text-text-muted">By class (AP @ tIoU)</div>
          <table className="mt-1 font-mono text-[10px] tabular-nums">
            <thead>
              <tr className="text-text-muted">
                <th className="py-0.5 text-left font-normal">Class</th>
                {bd.temporal.tolerances.map((t) => (
                  <th key={t} className={cn(numCell, 'font-normal')}>
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(bd.temporal.classes).map(([cls, aps]) => (
                <tr key={cls}>
                  <td className="py-0.5 pr-2 text-left text-text-secondary">{cls}</td>
                  {aps.map((v, i) => (
                    <td key={i} className={cn(numCell, 'text-text-secondary')}>
                      {pct(v)}
                    </td>
                  ))}
                </tr>
              ))}
              <tr className="border-t border-border text-text-primary">
                <td className="py-0.5 pr-2 text-left">overall</td>
                {bd.temporal.overall.map((v, i) => (
                  <td key={i} className={numCell}>
                    {pct(v)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
        {/* By video */}
        {bd.per_video && bd.per_video.length > 0 && (
          <div className="min-w-0 rounded-lg border border-border bg-surface-100 px-3 py-2.5">
            <div className="text-[9px] uppercase tracking-wider text-text-muted">By video</div>
            <table className="mt-1 w-full font-mono text-[10px] tabular-nums">
              <thead>
                <tr className="text-text-muted">
                  <th className="py-0.5 text-left font-normal">Video</th>
                  <th className="w-12 py-0.5 text-right font-normal">mAP</th>
                  <th className="w-14 py-0.5 text-right font-normal">rallies</th>
                </tr>
              </thead>
              <tbody>
                {[...bd.per_video].sort((a, b) => b.harmonic - a.harmonic).map((v) => (
                  <tr key={v.video}>
                    <td className="max-w-0 truncate py-0.5 pr-3 text-left text-text-secondary" title={v.video}>{v.video}</td>
                    <td className="py-0.5 text-right text-text-primary">{pct(v.harmonic)}</td>
                    <td className="py-0.5 text-right text-text-secondary">{v.events}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

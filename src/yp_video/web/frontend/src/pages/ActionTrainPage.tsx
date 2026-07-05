import { useEffect, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
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
import type { ActionMapBreakdown, ActionPerfData, ActionTrainProgress, ActionTrainStatus, Job } from '@/types/api';

interface Form {
  dataset: string;
  frame_dir: string;
  checkpoint_dir: string;
  init_checkpoint: string;
  resume_run: string; // '' = fresh run; else the exp/ save_dir to --resume
  feature_arch: string;
  temporal_arch: string;
  audio_backend: string;
  num_epochs: number;
  batch_size: number;
  clip_len: number;
  sample_fps: number;
  num_workers: number;
  gpu: number;
  learning_rate: number;
  warm_up_epochs: number;
  criterion: string;
  start_val_epoch: number;
  epoch_num_frames: number | '';
  training_mode: 'all' | 'split' | 'holdout';
  camera_view: 'all' | 'broadcast' | 'sideline';
  val_ratio: number;
  split_seed: number;
  predict_location: boolean;
  stop_vllm: boolean;
}

const BASE_FORM: Form = {
  dataset: 'yp_actions',
  frame_dir: '', // seeded from /action-train/status (the resolved ACTION_FRAMES_DIR)
  checkpoint_dir: '',
  init_checkpoint: '',
  resume_run: '',
  feature_arch: 'rny008_gsm',
  temporal_arch: 'gru',
  audio_backend: 'logmel',
  num_epochs: 50,
  batch_size: 8,
  clip_len: 64,
  sample_fps: 30,
  num_workers: 4,
  gpu: 0,
  learning_rate: 0.0003,
  warm_up_epochs: 3,
  criterion: 'map',
  start_val_epoch: 0,
  epoch_num_frames: '',
  training_mode: 'all',
  camera_view: 'all',
  val_ratio: 0.2,
  split_seed: 42,
  predict_location: true,
  stop_vllm: false,
};

const SELECTS = {
  feature_arch: ['rny008_gsm', 'rny002_gsm', 'convnextt_gsm', 'rn18_gsm'],
  temporal_arch: ['gru', 'deeper_gru', 'mstcn', 'asformer'],
  criterion: ['map', 'loss'],
} as const;

const NUM_FIELDS: Array<{ key: keyof Form; label: string; min?: number; max?: number; step?: number }> = [
  { key: 'num_epochs', label: 'Epochs', min: 1, max: 1000 },
  { key: 'batch_size', label: 'Batch', min: 1, max: 64 },
  { key: 'clip_len', label: 'Clip len', min: 8, max: 256 },
  { key: 'sample_fps', label: 'Sample fps', min: 0, max: 120 },
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

export function ActionTrainPage() {
  const [form, setForm] = useState<Form>(BASE_FORM);
  const [job, setJob] = useState<Job | null>(null);

  const statusQuery = useQuery({
    queryKey: ['action-train-status'],
    queryFn: () => apiFetch<ActionTrainStatus>(API.actionTrain.status),
    refetchInterval: job && !isTerminal(job.status) ? false : 20_000,
  });
  const status = statusQuery.data;

  // Per-epoch validation curve + per-video breakdown; refresh while training.
  const perfQuery = useQuery({
    queryKey: ['action-train-performance'],
    queryFn: () => apiFetch<ActionPerfData>(API.actionTrain.performance),
    refetchInterval: job && !isTerminal(job.status) ? 30_000 : false,
  });
  const perf = perfQuery.data;

  // Adopt any active job on first load.
  useEffect(() => {
    const active = status?.active_job;
    if (active && !job) setJob(active);
  }, [status?.active_job, job]);

  // Seed init_checkpoint from server options.
  useEffect(() => {
    const opts = status?.init_checkpoints ?? [];
    if (opts.length && !form.init_checkpoint) setForm((f) => ({ ...f, init_checkpoint: opts[0]!.value }));
  }, [status?.init_checkpoints, form.init_checkpoint]);

  // Seed frame_dir from the server's resolved ACTION_FRAMES_DIR.
  useEffect(() => {
    const fd = status?.action_annotations?.frame_dir;
    if (fd && !form.frame_dir) setForm((f) => ({ ...f, frame_dir: fd }));
  }, [status?.action_annotations?.frame_dir, form.frame_dir]);

  useSSE<Job>(job && !isTerminal(job.status) ? API.jobs.eventsSSE(job.id) : null, (data) => {
    setJob(data);
    if (isTerminal(data.status)) {
      if (data.status === 'completed') toast.success('Action training complete');
      if (data.status === 'failed') toast.error(`Action training failed: ${data.error || data.message}`);
    }
  });

  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));

  // Counts reflect the selected camera view (all = totals; otherwise the
  // per-view breakdown from the status endpoint).
  const ann = status?.action_annotations;
  const viewStats = form.camera_view === 'all' ? undefined : ann?.by_view?.[form.camera_view];
  const stats = {
    videos: Math.max(0, Number((viewStats ?? ann)?.videos) || 0),
    actions: Math.max(0, Number((viewStats ?? ann)?.events) || 0),
    frames: Math.max(0, Number((viewStats ?? ann)?.frames) || 0),
  };
  // Per-video action counts for the selected view, most-labelled first.
  const perVideo = (ann?.per_video ?? [])
    .filter((v) => form.camera_view === 'all' || v.view === form.camera_view)
    .slice()
    .sort((a, b) => b.events - a.events);
  const ready = stats.actions > 0;
  const running = !!job && (job.status === 'running' || job.status === 'pending');
  const canStart = !running && ready && Boolean(status?.spot_available);
  const showSplit = form.training_mode === 'split';
  const showHoldout = form.training_mode === 'holdout';

  const exportDataset = () => {
    if (!stats.actions) {
      toast.warning('No saved action annotations to export yet');
      return;
    }
    window.location.href = apiUrl(API.actionAnnotate.export);
  };

  const start = async () => {
    try {
      const body = {
        source: 'action_annotations',
        training_mode: form.training_mode,
        camera_view: form.camera_view,
        dataset: form.dataset.trim(),
        frame_dir: form.frame_dir.trim(),
        checkpoint_dir: form.checkpoint_dir.trim() || null,
        init_checkpoint: form.resume_run !== '' ? null : form.init_checkpoint.trim() || null,
        resume: form.resume_run !== '',
        save_dir: form.resume_run || null,
        gpu: form.gpu,
        feature_arch: form.feature_arch,
        temporal_arch: form.temporal_arch,
        audio_backend: form.audio_backend,
        clip_len: form.clip_len,
        sample_fps: form.sample_fps,
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
        predict_location: form.predict_location,
        stop_vllm: form.stop_vllm,
      };
      const started = await apiFetch<Job>(API.actionTrain.start, { method: 'POST', body });
      setJob(started);
      toast.success('Action training started');
    } catch (e) {
      toast.error(`Action training failed to start: ${errMsg(e)}`);
    }
  };

  const cancel = async () => {
    if (!job?.id) return;
    try {
      await apiFetch(API.jobs.cancel(job.id), { method: 'POST' });
      toast.warning('Action training cancelled');
    } catch (e) {
      toast.error(`Cancel failed: ${errMsg(e)}`);
    }
  };

  const initCheckpoints = status?.init_checkpoints ?? [];
  const resumableRuns = status?.resumable_runs ?? [];
  const isResuming = form.resume_run !== '';

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <Button size="sm" onClick={exportDataset}>
            Export JSONL
          </Button>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Action videos" value={stats.videos} tintClass="text-primary-light" />
        <StatTile label="Action labels" value={stats.actions} tintClass="text-primary-light" />
        <StatTile label="Action frames" value={stats.frames.toLocaleString()} tintClass="text-primary-light" />
        <StatTile label="Status" value={ready ? 'ready' : 'not ready'} tintClass={ready ? 'text-primary-light' : 'text-amber-400'} />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1.6fr_1fr]">
        {/* Training config */}
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
            <Field label="Dataset">
              <input value={form.dataset} onChange={(e) => set('dataset', e.target.value)} className={fieldCls} />
            </Field>
            <Field label="Init checkpoint" className="col-span-2">
              <select
                value={form.init_checkpoint}
                onChange={(e) => set('init_checkpoint', e.target.value)}
                title={isResuming ? 'Ignored while resuming (weights load from the run checkpoint)' : form.init_checkpoint}
                disabled={isResuming}
                className={cn(fieldCls, 'cursor-pointer appearance-none', isResuming && 'opacity-50')}
              >
                {initCheckpoints.length === 0 && <option value="">No checkpoints found</option>}
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
            <Field label="Audio">
              <select value={form.audio_backend} onChange={(e) => set('audio_backend', e.target.value)} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
                <option value="logmel">logmel (late fusion)</option>
                <option value="none">none (visual only)</option>
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

            <Field label="Data mode">
              <select value={form.training_mode} onChange={(e) => set('training_mode', e.target.value as Form['training_mode'])} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
                <option value="all">All Data</option>
                <option value="split">Train/Test Split</option>
                <option value="holdout">Holdout (val-set.txt)</option>
              </select>
            </Field>
            <Field label="Camera view">
              <select value={form.camera_view} onChange={(e) => set('camera_view', e.target.value as Form['camera_view'])} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
                <option value="all">All Views</option>
                <option value="broadcast">Broadcast</option>
                <option value="sideline">Sideline</option>
              </select>
            </Field>
            {showSplit && (
              <>
                <Field label="Val ratio">
                  <input type="number" value={form.val_ratio} min={0.01} max={0.9} step={0.01} onChange={(e) => set('val_ratio', Number(e.target.value))} className={cn(fieldCls, 'font-mono tabular-nums')} />
                </Field>
                <Field label="Split seed">
                  <input type="number" value={form.split_seed} onChange={(e) => set('split_seed', Number(e.target.value))} className={cn(fieldCls, 'font-mono tabular-nums')} />
                </Field>
              </>
            )}
          </div>

          {showHoldout && (
            <p className="mt-2 text-xs text-text-secondary">
              Validation videos are read from <code className="font-mono">videos/action-val-set.txt</code> — one filename per line. Every other labelled video trains.
            </p>
          )}

          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-text-secondary">
            <label className="inline-flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={form.predict_location} onChange={(e) => set('predict_location', e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Predict location
            </label>
            <label className="inline-flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={form.stop_vllm} onChange={(e) => set('stop_vllm', e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Stop vLLM
            </label>
          </div>

          <div className="mt-4 flex items-center gap-2">
            <Button intent="primary" onClick={start} disabled={!canStart} className="flex-1">
              {running ? 'Training…' : 'Start Training'}
            </Button>
            {running && (
              <Button onClick={cancel}>Cancel</Button>
            )}
          </div>
        </Card>

        {/* Dataset summary */}
        <Card>
          <SectionLabel>YP Action Labels</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {[
              ['Labels', `${stats.videos} vid / ${stats.actions} ev`],
              ['Frames', stats.frames.toLocaleString()],
              ['Mode', form.training_mode === 'all' ? 'all data' : form.training_mode === 'holdout' ? 'holdout (val-set.txt)' : 'train/test split'],
              ['View', form.camera_view === 'all' ? 'all views' : form.camera_view],
              ['Label dir', status?.action_annotations?.label_dir || '—'],
              ['Frame dir', form.frame_dir || status?.action_annotations?.frame_dir || '—'],
              ['Ckpt dir', form.checkpoint_dir || (status?.action_annotations?.checkpoint_dir ? `${status.action_annotations.checkpoint_dir}/<auto run>` : '—')],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-16 flex-shrink-0 text-text-muted">{label}</span>
                <span className="min-w-0 flex-1 truncate font-mono tabular-nums text-text-secondary" title={String(value)}>
                  {value}
                </span>
              </div>
            ))}
          </div>

          {perVideo.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-[11px] text-text-muted hover:text-text-primary">
                Per-video actions ({perVideo.length})
              </summary>
              <div className="mt-1.5 max-h-64 overflow-y-auto rounded-lg border border-border">
                <table className="w-full text-[10.5px] tabular-nums">
                  <thead className="sticky top-0 bg-surface-100 text-text-muted">
                    <tr>
                      <th className="px-2 py-1 text-left font-normal">Video</th>
                      <th className="px-2 py-1 text-right font-normal">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perVideo.map((v) => (
                      <tr key={v.video} className="border-t border-border/50">
                        <td className="max-w-0 truncate px-2 py-1 text-text-secondary" title={v.video}>
                          {v.is_val && (
                            <span className="mr-1 rounded bg-accent/20 px-1 text-[9px] font-semibold uppercase text-accent">val</span>
                          )}
                          {v.video}
                        </td>
                        <td className="px-2 py-1 text-right font-mono text-text-secondary">{v.events}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </Card>
      </div>

      {/* Training job */}
      {job && (
        <Card>
          <SectionLabel>Training job</SectionLabel>
          <JobProgress job={job} showLogs truncateMsg={false} />
          <TrainDetail progress={job.params?.action_train_progress as ActionTrainProgress | undefined} epochsFallback={form.num_epochs} />
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

function TrainDetail({ progress: p, epochsFallback }: { progress?: ActionTrainProgress; epochsFallback: number }) {
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
    ['Last mAP', latestMap],
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
          <summary className="cursor-pointer text-[10px] text-text-muted hover:text-text-primary">mAP breakdown</summary>
          {p.latest_val_breakdown && <MapBreakdownTable title={`Latest — Epoch ${p.epoch_display ?? 1}`} bd={p.latest_val_breakdown} />}
          {p.best_breakdown && <MapBreakdownTable title={`Best${p.best_epoch != null ? ` — Epoch ${p.best_epoch + 1}` : ''}`} bd={p.best_breakdown} />}
        </details>
      )}
    </>
  );
}

function MapBreakdownTable({ title, bd }: { title: string; bd: ActionMapBreakdown }) {
  const pct = (v: number | undefined) => (Number.isFinite(v) ? ((v as number) * 100).toFixed(1) : '—');
  return (
    <div className="mt-3">
      <div className="text-xs font-semibold text-text-primary">{title}</div>
      <table className="mt-1 w-full font-mono text-[10px] tabular-nums">
        <thead>
          <tr className="text-text-muted">
            <th className="text-left font-normal">Class</th>
            {bd.temporal.tolerances.map((t) => (
              <th key={t} className="text-right font-normal">
                tol{t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(bd.temporal.classes).map(([cls, aps]) => (
            <tr key={cls}>
              <td className="text-left text-text-secondary">{cls}</td>
              {aps.map((v, i) => (
                <td key={i} className="text-right text-text-secondary">
                  {pct(v)}
                </td>
              ))}
            </tr>
          ))}
          <tr className="border-t border-border text-text-primary">
            <td className="text-left">overall</td>
            {bd.temporal.overall.map((v, i) => (
              <td key={i} className="text-right">
                {pct(v)}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      <div className="mt-1 text-[10px] text-text-secondary">
        spatial
        {bd.spatial.pixel_tolerances.map((px, i) => ` ${px}px ${pct(bd.spatial.overall_by_px[i])}`).join('')}
        {' · overall '}
        {pct(bd.spatial.overall)}
      </div>
      {bd.per_video && bd.per_video.length > 0 && (
        <table className="mt-2 w-full font-mono text-[10px] tabular-nums">
          <thead>
            <tr className="text-text-muted">
              <th className="text-left font-normal">Video</th>
              <th className="text-right font-normal">mAP</th>
              <th className="text-right font-normal">temp</th>
              <th className="text-right font-normal">spat</th>
            </tr>
          </thead>
          <tbody>
            {[...bd.per_video].sort((a, b) => b.harmonic - a.harmonic).map((v) => (
              <tr key={v.video}>
                <td className="max-w-0 truncate text-left text-text-secondary" title={v.video}>{v.video}</td>
                <td className="text-right text-text-primary">{pct(v.harmonic)}</td>
                <td className="text-right text-text-secondary">{pct(v.temporal)}</td>
                <td className="text-right text-text-secondary">{pct(v.spatial)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

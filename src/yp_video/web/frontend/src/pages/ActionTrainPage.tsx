import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { CameraViewSelect, Field, InitCheckpointSelect, SelectArch, fieldCls } from '@/components/train/Field';
import { useTrainPerformance } from '@/components/train/useTrainPerformance';
import { useSingleJob } from '@/lib/useSingleJob';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { TrainJobCard } from '@/components/train/TrainJobCard';
import { TrainPerfCard } from '@/components/train/TrainPerfCard';
import { toast } from '@/components/feedback/toast';
import type { ActionTrainStatus, Job } from '@/types/api';

interface Form {
  dataset: string;
  frame_dir: string;
  checkpoint_dir: string;
  init_checkpoint: string;
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
  camera_view: 'all' | 'broadcast' | 'sideline';
  predict_location: boolean;
  stop_vllm: boolean;
}

const BASE_FORM: Form = {
  dataset: 'yp_actions',
  frame_dir: '', // seeded from /action-train/status (the resolved ACTION_FRAMES_DIR)
  checkpoint_dir: '',
  init_checkpoint: '',
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
  camera_view: 'all',
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

export function ActionTrainPage() {
  const [form, setForm] = useState<Form>(BASE_FORM);
  const [holdoutVideos, setHoldoutVideos] = useState<Set<string>>(new Set());
  const holdoutSeeded = useRef(false);

  const statusQuery = useQuery({
    queryKey: ['action-train-status'],
    queryFn: () => apiFetch<ActionTrainStatus>(API.actionTrain.status),
  });
  const status = statusQuery.data;

  const { job, setJob, running, cancel } = useSingleJob({
    activeJob: status?.active_job,
    label: 'Action training',
  });
  // Per-epoch validation curve + per-video breakdown; refresh while training.
  const { perf, setPerfRun } = useTrainPerformance(
    'action-train-performance',
    API.actionTrain.performance,
    running,
  );

  // Seed frame_dir from the server's resolved ACTION_FRAMES_DIR.
  useEffect(() => {
    const fd = status?.action_annotations?.frame_dir;
    if (fd && !form.frame_dir) setForm((f) => ({ ...f, frame_dir: fd }));
  }, [status?.action_annotations?.frame_dir, form.frame_dir]);

  // Seed the holdout selection from the saved val-set flags, once.
  useEffect(() => {
    const perVideoAll = status?.action_annotations?.per_video;
    if (holdoutSeeded.current || !perVideoAll?.length) return;
    setHoldoutVideos(new Set(perVideoAll.filter((v) => v.is_val).map((v) => v.video)));
    holdoutSeeded.current = true;
  }, [status?.action_annotations?.per_video]);

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
  // Validation is always a manual per-video holdout; the list is the same
  // view-filtered corpus the run trains on.
  const holdoutChoices = perVideo;
  const selectedHoldoutCount = holdoutChoices.filter((v) => holdoutVideos.has(v.video)).length;
  const ready = stats.actions > 0;
  const canStart = !running && ready && Boolean(status?.spot_available) && selectedHoldoutCount > 0;

  const start = async () => {
    try {
      const body = {
        source: 'action_annotations',
        training_mode: 'holdout',
        holdout_videos: holdoutChoices
          .filter((v) => holdoutVideos.has(v.video))
          .map((v) => `${v.video}_actions.jsonl`),
        camera_view: form.camera_view,
        dataset: form.dataset.trim(),
        frame_dir: form.frame_dir.trim(),
        checkpoint_dir: form.checkpoint_dir.trim() || null,
        init_checkpoint: form.init_checkpoint.trim() || null,
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

  const initCheckpoints = status?.init_checkpoints ?? [];

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {/* Training config */}
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
            <Field label="Dataset">
              <input value={form.dataset} onChange={(e) => set('dataset', e.target.value)} className={fieldCls} />
            </Field>
            <InitCheckpointSelect
              value={form.init_checkpoint}
              onChange={(v) => set('init_checkpoint', v)}
              options={initCheckpoints}
            />
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
            <CameraViewSelect value={form.camera_view} onChange={(v) => set('camera_view', v)} />
          </div>

          <div className="mt-5 border-t border-border pt-4">
            <SectionLabel>Validation split</SectionLabel>
            <div className="mb-2 flex items-center justify-between text-[11px] text-text-muted">
              <span>
                {selectedHoldoutCount} validation / {holdoutChoices.length} in this camera view
              </span>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() =>
                    setHoldoutVideos(
                      new Set((ann?.per_video ?? []).filter((v) => v.is_val).map((v) => v.video)),
                    )
                  }
                  className="text-primary-light hover:underline"
                >
                  Load saved val set
                </button>
                <button
                  type="button"
                  onClick={() => setHoldoutVideos(new Set())}
                  className="text-text-muted hover:underline"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="max-h-64 space-y-1 overflow-auto rounded-lg border border-border bg-surface-50 p-2">
              {holdoutChoices.map((video) => (
                <label
                  key={video.video}
                  className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 hover:bg-ink/[0.03]"
                >
                  <input
                    type="checkbox"
                    checked={holdoutVideos.has(video.video)}
                    onChange={(e) => {
                      const next = new Set(holdoutVideos);
                      if (e.target.checked) next.add(video.video);
                      else next.delete(video.video);
                      setHoldoutVideos(next);
                    }}
                    className="h-3.5 w-3.5 flex-shrink-0 accent-primary"
                  />
                  <span className="min-w-0 flex-1 truncate text-xs text-text-secondary">
                    {video.video}
                  </span>
                  <Badge tone="neutral">{video.view}</Badge>
                  <span className="font-mono text-[10px] tabular-nums text-text-muted">
                    {video.events}
                  </span>
                </label>
              ))}
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
              Selected videos are validation only; every other matching action annotation trains.
            </p>
          </div>

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
              ['Mode', 'manual holdout'],
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
      <TrainJobCard
        job={job}
        progressKey="action_train_progress"
        epochsFallback={form.num_epochs}
        onCancel={() => void cancel()}
      />

      {/* Per-epoch curve + per-video mAP for the selected (or latest) run */}
      {perf && perf.entries.length > 0 && <TrainPerfCard data={perf} onSelectRun={setPerfRun} />}
    </div>
  );
}

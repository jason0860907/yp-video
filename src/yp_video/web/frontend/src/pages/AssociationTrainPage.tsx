import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { toast } from '@/components/feedback/toast';
import { JobProgress } from '@/components/job/JobProgress';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { API, apiFetch, errMsg } from '@/lib/api';
import { fieldCls } from '@/components/train/Field';
import { cn } from '@/lib/cn';
import { useSingleJob } from '@/lib/useSingleJob';
import type {
  AssociationVideo,
  Job,
  JobLogs,
  ReidAssociationStatus,
} from '@/types/api';

type Form = {
  run_name: string;
  init_checkpoint: string;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  backbone_learning_rate: number;
  warm_up_epochs: number;
  gpu: number;
  backbone: 'rny002' | 'rny002_gsm' | 'rny008' | 'rny008_gsm' | 'rn18' | 'rn50';
  crop_dim: number;
  num_workers: number;
  stop_vllm: boolean;
};

type LiveAssociationMetrics = {
  epoch_display?: number;
  epochs?: number;
  train_loss?: number;
  val_loss?: number;
  train?: AssociationMetricSet;
  val?: AssociationMetricSet;
  best?: boolean;
};

type AssociationMetricSet = {
    player_top1?: number | null;
    player_coverage?: number | null;
    selective_accuracy?: number | null;
    overall_exact?: number | null;
    occluded_recall?: number | null;
    untracked_recall?: number | null;
};

type AssociationHistoryPoint = {
  epoch: number;
  train_player_top1?: number | null;
  val_player_top1?: number | null;
  train_overall_exact?: number | null;
  val_overall_exact?: number | null;
  train_loss?: number | null;
  val_loss?: number | null;
  best?: boolean;
};

type AssociationHistoryResponse = {
  run: string | null;
  history: AssociationHistoryPoint[];
};

const INITIAL_FORM: Form = {
  run_name: '',
  init_checkpoint: '',
  num_epochs: 40,
  batch_size: 8,
  learning_rate: 0.0003,
  backbone_learning_rate: 0.00003,
  warm_up_epochs: 3,
  gpu: 0,
  backbone: 'rny002',
  crop_dim: 224,
  num_workers: 4,
  stop_vllm: false,
};

export function AssociationTrainPage() {
  const initialized = useRef(false);
  const startingRef = useRef(false);
  const [trainVideos, setTrainVideos] = useState<Set<string>>(new Set());
  const [valVideos, setValVideos] = useState<Set<string>>(new Set());
  const [form, setForm] = useState<Form>(INITIAL_FORM);
  const [starting, setStarting] = useState(false);
  const [metricHistory, setMetricHistory] = useState<AssociationHistoryPoint[]>([]);

  const videosQuery = useQuery({
    queryKey: ['association-videos'],
    queryFn: () => apiFetch<AssociationVideo[]>(API.association.videos),
  });
  const statusQuery = useQuery({
    queryKey: ['actor-association-status'],
    queryFn: () => apiFetch<ReidAssociationStatus>(API.association.status),
  });

  const videos = videosQuery.data ?? [];
  const status = statusQuery.data;

  const { job, setJob, running: training } = useSingleJob({
    activeJob: status?.active_job,
    label: 'yp-association training',
  });
  const historyQuery = useQuery({
    queryKey: ['association-train-history', job?.params?.save_dir],
    queryFn: () => apiFetch<AssociationHistoryResponse>(API.association.trainHistory),
    enabled: Boolean(job),
    refetchInterval: training ? 5_000 : false,
    retry: false,
  });
  const logsQuery = useQuery({
    queryKey: ['association-train-logs', job?.id],
    queryFn: () => apiFetch<JobLogs>(API.jobs.logs(job!.id)),
    enabled: Boolean(job?.id),
    refetchInterval: training ? 3_000 : false,
  });

  useEffect(() => {
    if (initialized.current || videosQuery.data === undefined) return;
    const ready = videosQuery.data.filter(
      (video) => video.event_count > 0 && video.unreviewed === 0,
    );
    const valCount = ready.length >= 2
      ? Math.max(1, Math.round(ready.length * 0.2))
      : 0;
    const splitAt = ready.length - valCount;
    setTrainVideos(new Set(ready.slice(0, splitAt).map((video) => video.name)));
    setValVideos(new Set(ready.slice(splitAt).map((video) => video.name)));
    initialized.current = true;
  }, [videosQuery.data]);

  useEffect(() => {
    const fromApi = historyQuery.data?.history ?? [];
    const fromLogs = historyFromLogs(logsQuery.data?.lines ?? []);
    const latest = historyPoint(
      job?.params?.association_train_progress as LiveAssociationMetrics | undefined,
    );
    setMetricHistory((previous) =>
      mergeHistory(previous, fromApi, fromLogs, latest ? [latest] : []),
    );
  }, [
    historyQuery.data,
    job?.params?.association_train_progress,
    logsQuery.data,
  ]);

  const set = <K extends keyof Form>(key: K, value: Form[K]) =>
    setForm((previous) => ({ ...previous, [key]: value }));

  const chooseTrain = (next: Set<string>) => {
    setTrainVideos(next);
    setValVideos((previous) => {
      const filtered = new Set(previous);
      for (const name of next) filtered.delete(name);
      return filtered;
    });
  };

  const chooseValidation = (next: Set<string>) => {
    setValVideos(next);
    setTrainVideos((previous) => {
      const filtered = new Set(previous);
      for (const name of next) filtered.delete(name);
      return filtered;
    });
  };

  const counts = useMemo(() => {
    const sum = (selected: Set<string>) =>
      videos
        .filter((video) => selected.has(video.name))
        .reduce((total, video) => total + video.reviewed, 0);
    return {
      train: sum(trainVideos),
      validation: sum(valVideos),
    };
  }, [trainVideos, valVideos, videos]);

  const busy = starting || training;
  const canTrain =
    !busy
    && status?.spot_available === true
    && trainVideos.size > 0
    && valVideos.size > 0
    && counts.train > 0
    && counts.validation > 0;

  const start = async () => {
    if (startingRef.current) return;
    if (!canTrain) {
      toast.warning('Select reviewed training and validation videos');
      return;
    }
    startingRef.current = true;
    setStarting(true);
    toast.info('Validating the selected videos and creating the training job…');
    try {
      const started = await apiFetch<Job>(API.association.train, {
        method: 'POST',
        body: {
          train_videos: [...trainVideos],
          val_videos: [...valVideos],
          run_name: form.run_name.trim() || null,
          init_checkpoint: form.init_checkpoint || null,
          num_epochs: form.num_epochs,
          batch_size: form.batch_size,
          learning_rate: form.learning_rate,
          backbone_learning_rate: form.backbone_learning_rate,
          warm_up_epochs: form.warm_up_epochs,
          gpu: form.gpu,
          backbone: form.backbone,
          crop_dim: form.crop_dim,
          num_workers: form.num_workers,
          stop_vllm: form.stop_vllm,
        },
      });
      setJob(started);
      toast.success('yp-association training started');
    } catch (error) {
      toast.error(`Training failed to start: ${errMsg(error)}`);
    } finally {
      startingRef.current = false;
      setStarting(false);
    }
  };
  const live = job?.params?.association_train_progress as
    | LiveAssociationMetrics
    | undefined;
  const percent = (value: number | null | undefined) =>
    value == null ? '—' : `${(value * 100).toFixed(1)}%`;

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <Button intent="primary" disabled={!canTrain} onClick={() => void start()}>
            {starting ? 'Starting…' : training ? 'Training…' : 'Train yp-association'}
          </Button>
        }
      />

      <Card>
        <SectionLabel>yp-association · independent event model</SectionLabel>
        <p className="mb-4 max-w-4xl text-xs leading-relaxed text-text-muted">
          One reviewed action event is one training example. The model reads
          nine synchronized frames, the contact point and every candidate
          track, then chooses a player, occluded or untracked. It has its own
          dataset, optimizer, validation metrics and checkpoint; action
          classification and location loss are not part of this run.
        </p>

        {status?.spot_available === false ? (
          <p className="mb-4 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-400">
            yp-spot or its Python environment is unavailable on this machine.
          </p>
        ) : null}

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <VideoPicker
            title="Training videos"
            subtitle={`${trainVideos.size} videos · ${counts.train} reviewed events`}
            videos={videos}
            selected={trainVideos}
            onChange={chooseTrain}
            exclude={valVideos}
          />
          <VideoPicker
            title="Validation videos"
            subtitle={`${valVideos.size} videos · ${counts.validation} reviewed events`}
            videos={videos}
            selected={valVideos}
            onChange={chooseValidation}
            exclude={trainVideos}
          />
        </div>
      </Card>

      <Card>
        <SectionLabel>Training configuration</SectionLabel>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="Run name" className="col-span-2">
            <input
              value={form.run_name}
              onChange={(event) => set('run_name', event.target.value)}
              placeholder="yp_actor_YYYYMMDD-HHMMSS"
              className={fieldCls}
            />
          </Field>
          <Field label="Init checkpoint" className="col-span-2">
            <select
              value={form.init_checkpoint}
              onChange={(event) => set('init_checkpoint', event.target.value)}
              className={cn(fieldCls, 'cursor-pointer')}
            >
              <option value="">New model (ImageNet-initialized backbone)</option>
              {(status?.init_checkpoints ?? []).map((checkpoint) => (
                <option key={checkpoint.value} value={checkpoint.value}>
                  {checkpoint.label}
                </option>
              ))}
            </select>
          </Field>
          <NumberField label="Epochs" value={form.num_epochs} min={1} onChange={(value) => set('num_epochs', value)} />
          <NumberField label="Batch" value={form.batch_size} min={1} onChange={(value) => set('batch_size', value)} />
          <NumberField label="Head learning rate" value={form.learning_rate} min={0} step={0.00001} onChange={(value) => set('learning_rate', value)} />
          <NumberField label="Backbone learning rate" value={form.backbone_learning_rate} min={0} step={0.00001} onChange={(value) => set('backbone_learning_rate', value)} />
          <NumberField label="Warm-up epochs" value={form.warm_up_epochs} min={0} onChange={(value) => set('warm_up_epochs', value)} />
          <NumberField label="GPU" value={form.gpu} min={0} max={7} onChange={(value) => set('gpu', value)} />
          <Field label="Visual backbone">
            <select
              value={form.backbone}
              onChange={(event) => set('backbone', event.target.value as Form['backbone'])}
              className={cn(fieldCls, 'cursor-pointer')}
            >
              {['rny002', 'rny002_gsm', 'rny008', 'rny008_gsm', 'rn18', 'rn50'].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </Field>
          <NumberField label="Image size" value={form.crop_dim} min={64} max={512} onChange={(value) => set('crop_dim', value)} />
          <NumberField label="Workers" value={form.num_workers} min={0} onChange={(value) => set('num_workers', value)} />
        </div>

        <label className="mt-4 inline-flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
          <input
            type="checkbox"
            checked={form.stop_vllm}
            onChange={(event) => set('stop_vllm', event.target.checked)}
            className="h-3.5 w-3.5 accent-primary"
          />
          Stop vLLM before taking the GPU
        </label>

        <div className="mt-4 flex items-center gap-3">
          <Button intent="primary" disabled={!canTrain} onClick={() => void start()}>
            {starting ? 'Starting…' : training ? 'Training…' : 'Train yp-association'}
          </Button>
          {!canTrain && !busy ? (
            <span className="text-[11px] text-amber-400">
              Keep at least one reviewed video in each split.
            </span>
          ) : null}
        </div>
      </Card>

      {job ? (
        <Card>
          <SectionLabel>Training job</SectionLabel>
          <JobProgress job={job} />
          {live?.val ? (
            <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
              <Metric label="Player Top-1" value={percent(live.val.player_top1)} />
              <Metric label="Player coverage" value={percent(live.val.player_coverage)} />
              <Metric label="Overall exact" value={percent(live.val.overall_exact)} />
              <Metric label="Occluded recall" value={percent(live.val.occluded_recall)} />
              <Metric label="Untracked recall" value={percent(live.val.untracked_recall)} />
              <Metric label="Validation loss" value={live.val_loss?.toFixed(4) ?? '—'} />
              <Metric label="Training loss" value={live.train_loss?.toFixed(4) ?? '—'} />
              <Metric label="Epoch" value={`${live.epoch_display ?? '—'} / ${live.epochs ?? '—'}`} />
            </div>
          ) : null}
          {metricHistory.length ? (
            <AccuracyChart points={metricHistory} />
          ) : null}
        </Card>
      ) : null}

      <Card>
        <SectionLabel>Available association checkpoints</SectionLabel>
        {status?.association_checkpoints.length ? (
          <div className="space-y-2">
            {status.association_checkpoints.map((checkpoint) => (
              <div
                key={checkpoint.path}
                className="rounded-lg border border-border bg-surface-50 px-3 py-2 text-xs"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-text-primary">{checkpoint.name}</span>
                  <Badge tone="success">Association Predict ready</Badge>
                  {checkpoint.family === 'legacy-actor-head' ? (
                    <Badge tone="neutral">fusion actor head</Badge>
                  ) : null}
                  {checkpoint.epoch != null ? (
                    <span className="text-text-muted">epoch {checkpoint.epoch + 1}</span>
                  ) : null}
                </div>
                {checkpoint.family === 'legacy-actor-head' ? (
                  <div className="mt-1 flex flex-wrap gap-3 font-mono text-[11px] tabular-nums text-text-secondary">
                    <span>overall Top-1 {checkpoint.metrics.all_top1 == null ? '—' : `${(checkpoint.metrics.all_top1 * 100).toFixed(1)}%`}</span>
                    <span>hard Top-1 {checkpoint.metrics.hard_top1 == null ? '—' : `${(checkpoint.metrics.hard_top1 * 100).toFixed(1)}%`}</span>
                    <span>manual Top-1 {checkpoint.metrics.manual_top1 == null ? '—' : `${(checkpoint.metrics.manual_top1 * 100).toFixed(1)}%`}</span>
                  </div>
                ) : (
                  <div className="mt-1 flex flex-wrap gap-3 font-mono text-[11px] tabular-nums text-text-secondary">
                    <span>player Top-1 {checkpoint.metrics.player_top1 == null ? '—' : `${(checkpoint.metrics.player_top1 * 100).toFixed(1)}%`}</span>
                    <span>overall {checkpoint.metrics.overall_exact == null ? '—' : `${(checkpoint.metrics.overall_exact * 100).toFixed(1)}%`}</span>
                    <span>coverage {checkpoint.metrics.player_coverage == null ? '—' : `${(checkpoint.metrics.player_coverage * 100).toFixed(1)}%`}</span>
                  </div>
                )}
                <p className="mt-1 break-all text-[10px] text-text-muted">
                  {checkpoint.path}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-text-muted">
            No association checkpoint yet.
          </p>
        )}
      </Card>
    </div>
  );
}

function historyPoint(
  live: LiveAssociationMetrics | undefined,
): AssociationHistoryPoint | null {
  if (live?.epoch_display == null) return null;
  return {
    epoch: live.epoch_display,
    train_player_top1: live.train?.player_top1,
    val_player_top1: live.val?.player_top1,
    train_overall_exact: live.train?.overall_exact,
    val_overall_exact: live.val?.overall_exact,
    train_loss: live.train_loss,
    val_loss: live.val_loss,
    best: live.best,
  };
}

function historyFromLogs(lines: string[]): AssociationHistoryPoint[] {
  const prefix = 'ASSOCIATION_METRICS ';
  return lines.flatMap((line) => {
    const index = line.indexOf(prefix);
    if (index < 0) return [];
    try {
      const record = JSON.parse(line.slice(index + prefix.length)) as {
        epoch?: number;
        train?: LiveAssociationMetrics['train'];
        val?: LiveAssociationMetrics['val'];
        loss?: { train?: number; val?: number };
        best?: boolean;
      };
      if (record.epoch == null) return [];
      return [{
        epoch: record.epoch + 1,
        train_player_top1: record.train?.player_top1,
        val_player_top1: record.val?.player_top1,
        train_overall_exact: record.train?.overall_exact,
        val_overall_exact: record.val?.overall_exact,
        train_loss: record.loss?.train,
        val_loss: record.loss?.val,
        best: record.best,
      }];
    } catch {
      return [];
    }
  });
}

function mergeHistory(...groups: AssociationHistoryPoint[][]): AssociationHistoryPoint[] {
  const byEpoch = new Map<number, AssociationHistoryPoint>();
  for (const point of groups.flat()) {
    byEpoch.set(point.epoch, { ...byEpoch.get(point.epoch), ...point });
  }
  return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
}

function AccuracyChart({ points }: { points: AssociationHistoryPoint[] }) {
  const width = 900;
  const height = 300;
  const margin = { left: 52, right: 18, top: 20, bottom: 42 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const maxEpoch = Math.max(points.at(-1)?.epoch ?? 1, 2);
  const x = (epoch: number) =>
    margin.left + ((epoch - 1) / (maxEpoch - 1)) * innerWidth;
  const y = (value: number) =>
    margin.top + (1 - Math.max(0, Math.min(1, value))) * innerHeight;
  const path = (key: 'train_player_top1' | 'val_player_top1') =>
    points
      .filter((point) => point[key] != null)
      .map((point, index) =>
        `${index ? 'L' : 'M'} ${x(point.epoch).toFixed(1)} ${y(point[key]!).toFixed(1)}`,
      )
      .join(' ');
  const validation = points.filter((point) => point.val_player_top1 != null);
  const best = validation.reduce<AssociationHistoryPoint | null>(
    (winner, point) =>
      winner == null || point.val_player_top1! > winner.val_player_top1!
        ? point
        : winner,
    null,
  );

  return (
    <div className="mt-4 rounded-lg border border-border bg-surface-50 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs font-medium text-text-primary">
            Player Top-1 by epoch
          </p>
          <p className="text-[10px] text-text-muted">
            選對球員的比例；train 與 validation 的距離代表 generalization gap
          </p>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-text-secondary">
          <Legend color="#94a3b8" label="Train" />
          <Legend color="#22d3ee" label="Validation" />
          <span className="font-mono">
            Best {best ? `E${best.epoch} · ${(best.val_player_top1! * 100).toFixed(1)}%` : '—'}
          </span>
        </div>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Training and validation Player Top-1 accuracy by epoch"
        className="h-auto w-full"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={margin.left}
              x2={width - margin.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="currentColor"
              className="text-border"
              strokeWidth="1"
            />
            <text
              x={margin.left - 9}
              y={y(tick) + 4}
              textAnchor="end"
              className="fill-text-muted text-[10px]"
            >
              {Math.round(tick * 100)}%
            </text>
          </g>
        ))}
        <path
          d={path('train_player_top1')}
          fill="none"
          stroke="#94a3b8"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
        <path
          d={path('val_player_top1')}
          fill="none"
          stroke="#22d3ee"
          strokeWidth="2.5"
          vectorEffect="non-scaling-stroke"
        />
        {points.map((point) => (
          <g key={point.epoch}>
            {point.train_player_top1 != null ? (
              <circle
                cx={x(point.epoch)}
                cy={y(point.train_player_top1)}
                r="3"
                fill="#94a3b8"
              >
                <title>{`Epoch ${point.epoch} train ${(point.train_player_top1 * 100).toFixed(1)}%`}</title>
              </circle>
            ) : null}
            {point.val_player_top1 != null ? (
              <circle
                cx={x(point.epoch)}
                cy={y(point.val_player_top1)}
                r={point.epoch === best?.epoch ? 5 : 3.5}
                fill="#22d3ee"
                stroke={point.epoch === best?.epoch ? '#ecfeff' : 'none'}
                strokeWidth="2"
              >
                <title>{`Epoch ${point.epoch} validation ${(point.val_player_top1 * 100).toFixed(1)}%`}</title>
              </circle>
            ) : null}
          </g>
        ))}
        {points.map((point) => (
          <text
            key={`epoch-${point.epoch}`}
            x={x(point.epoch)}
            y={height - 15}
            textAnchor="middle"
            className="fill-text-muted text-[9px]"
          >
            {point.epoch}
          </text>
        ))}
        <text
          x={margin.left + innerWidth / 2}
          y={height - 1}
          textAnchor="middle"
          className="fill-text-muted text-[10px]"
        >
          Epoch
        </text>
      </svg>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="h-0.5 w-4" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface-50 px-3 py-2">
      <p className="text-[10px] text-text-muted">{label}</p>
      <p className="font-mono text-sm tabular-nums text-text-primary">{value}</p>
    </div>
  );
}

function VideoPicker({
  title,
  subtitle,
  videos,
  selected,
  onChange,
  exclude,
}: {
  title: string;
  subtitle: string;
  videos: AssociationVideo[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
  exclude: Set<string>;
}) {
  const available = videos.filter((video) => !exclude.has(video.name));
  return (
    <div className="rounded-lg border border-border p-3">
      <p className="mb-2 text-[11px] text-text-muted">{subtitle}</p>
      <VideoMultiSelectList
        videos={available}
        selected={selected}
        onSelectedChange={onChange}
        title={title}
        statusOptions={[
          { value: 'all', label: 'All', predicate: () => true },
          {
            value: 'done',
            label: 'Done',
            predicate: (video) => video.event_count > 0 && video.unreviewed === 0,
          },
          {
            value: 'partial',
            label: 'In progress',
            predicate: (video) => video.reviewed > 0 && video.unreviewed > 0,
          },
        ]}
        quickSelects={[
          {
            label: 'Done only',
            predicate: (video) => video.event_count > 0 && video.unreviewed === 0,
          },
          { label: 'Clear', predicate: () => false },
        ]}
        renderMeta={(video) => (
          <>
            <span className="font-mono text-[11px] tabular-nums text-text-muted">
              {video.reviewed}/{video.event_count}
            </span>
            {video.unreviewed === 0 && video.event_count > 0 ? (
              <Badge tone="success">Done</Badge>
            ) : video.reviewed > 0 ? (
              <Badge tone="warning">{video.unreviewed} left</Badge>
            ) : (
              <Badge>Unlabeled</Badge>
            )}
          </>
        )}
        maxHeightClass="max-h-[42vh]"
        emptyTitle="No association videos"
        emptySubtitle="Finish tracking, actions and Association Label first"
      />
    </div>
  );
}

function Field({
  label,
  className,
  children,
}: {
  label: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <label className={cn('block text-xs text-text-secondary', className)}>
      <span className="mb-1 block">{label}</span>
      {children}
    </label>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        className={cn(fieldCls, 'font-mono tabular-nums')}
      />
    </Field>
  );
}

import { useEffect, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { useSSE } from '@/lib/useSSE';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { TriStateFilter, type TriState } from '@/components/ui/TriStateFilter';
import { JobProgressCard } from '@/components/job/JobProgressCard';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { Job, PredictVideo, TrainCheckpoint, VllmStatus } from '@/types/api';

type KindFilter = 'all' | 'broadcast' | 'sideline';
type FeatureModel = 'base' | 'large' | 'giant' | 'gigantic';
type Row = PredictVideo & { selected: boolean };
type FilterKey = 'annotated' | 'pre_annotated' | 'features' | 'prediction';

const MODELS: Array<{ value: FeatureModel; label: string }> = [
  { value: 'base', label: 'ViT-B (768d)' },
  { value: 'large', label: 'ViT-L (1024d)' },
  { value: 'giant', label: 'ViT-g (1408d)' },
  { value: 'gigantic', label: 'ViT-G (1664d)' },
];
const KIND_TABS: Array<{ key: KindFilter; label: string }> = [
  { key: 'all', label: 'All' },
  { key: 'broadcast', label: 'Broadcast' },
  { key: 'sideline', label: 'Sideline' },
];
const FILTER_FIELDS: ReadonlyArray<{ key: FilterKey; label: string }> = [
  { key: 'annotated', label: 'annotated' },
  { key: 'pre_annotated', label: 'pre-ann' },
  { key: 'features', label: 'features' },
  { key: 'prediction', label: 'predicted' },
];
const CKPT_TAG: Record<TrainCheckpoint['kind'], string> = { best: '★', last: '◆', epoch: '' };

const fieldCls =
  'w-full rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';
const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const isTerminal = (s: Job['status']) => s === 'completed' || s === 'failed' || s === 'cancelled';

export function PredictPage() {
  const qc = useQueryClient();
  const [videos, setVideos] = useState<Row[]>([]);
  const [model, setModel] = useState<FeatureModel>('base');
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [filter, setFilter] = useState<Record<FilterKey, TriState>>({
    annotated: null,
    pre_annotated: null,
    features: null,
    prediction: null,
  });
  const [checkpoint, setCheckpoint] = useState('');
  const [showAllCkpts, setShowAllCkpts] = useState(false);
  const [threshold, setThreshold] = useState(0.3);
  const [device, setDevice] = useState<'cuda' | 'cpu'>('cuda');
  const [cutRallies, setCutRallies] = useState(false);
  const [trimActions, setTrimActions] = useState(true);
  const [job, setJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);

  const videosQuery = useQuery({
    queryKey: ['predict-videos'],
    queryFn: () => apiFetch<PredictVideo[]>(API.predict.videos),
  });
  const ckptQuery = useQuery({
    queryKey: ['train-checkpoints', showAllCkpts],
    queryFn: () => apiFetch<TrainCheckpoint[]>(API.train.checkpoints({ show_all: showAllCkpts })),
  });

  useEffect(() => {
    if (videosQuery.data) setVideos(videosQuery.data.map((v) => ({ ...v, selected: !v.has_prediction })));
  }, [videosQuery.data]);

  useSSE<Job>(running && job ? API.jobs.eventsSSE(job.id) : null, (data) => {
    setJob(data);
    if (isTerminal(data.status)) {
      setRunning(false);
      if (data.status === 'failed') toast.error(`Prediction failed: ${data.error || 'Unknown error'}`);
      else toast.success(data.message || 'Prediction complete!');
      void qc.invalidateQueries({ queryKey: ['predict-videos'] });
    }
  });

  const has = (v: Row, key: FilterKey) => {
    if (key === 'annotated') return !!v.has_annotation;
    if (key === 'pre_annotated') return !!v.has_pre_annotation;
    if (key === 'prediction') return !!v.has_prediction;
    return !!v.features?.[model];
  };
  const matches = (v: Row) => {
    if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
    return (Object.keys(filter) as FilterKey[]).every((k) => filter[k] === null || has(v, k) === filter[k]);
  };

  const visible = videos.filter(matches);
  const counts = {
    all: videos.length,
    broadcast: videos.filter((v) => v.kind === 'broadcast').length,
    sideline: videos.filter((v) => v.kind === 'sideline').length,
  };
  const selectedTotal = videos.filter((v) => v.selected).length;
  const predictedTotal = videos.filter((v) => v.has_prediction).length;
  const allVisibleSelected = visible.length > 0 && visible.every((v) => v.selected);

  const setVisibleSelection = (on: boolean) =>
    setVideos((prev) => prev.map((v) => (matches(v) ? { ...v, selected: on } : v)));

  const start = async () => {
    const selected = videos.filter((v) => v.selected).map((v) => v.name);
    if (selected.length === 0) return toast.warning('No videos selected');
    if (!checkpoint) return toast.warning('Select a checkpoint');

    let stopVllm = false;
    const vllm = await apiFetch<VllmStatus>(API.system.vllmStatus).catch(() => null);
    if (vllm?.status === 'running') {
      const ok = await confirm({
        title: 'Stop vLLM for prediction?',
        body: 'vLLM holds most of the GPU VRAM; prediction needs it for V-JEPA and would OOM otherwise. vLLM restarts automatically once prediction finishes.',
        confirmText: 'Stop & Predict',
        variant: 'warning',
      });
      if (!ok) return;
      stopVllm = true;
    }

    setRunning(true);
    try {
      const started = await apiFetch<Job>(API.predict.start, {
        method: 'POST',
        body: {
          videos: selected,
          checkpoint,
          threshold,
          device,
          cut_rallies: cutRallies,
          model,
          stop_vllm: stopVllm,
          trim_with_actions: trimActions,
        },
      });
      setJob(started);
    } catch (e) {
      setRunning(false);
      toast.error(`Failed to start prediction: ${errMsg(e)}`);
    }
  };

  const checkpoints = ckptQuery.data ?? [];
  const pct = Math.round((job?.progress ?? 0) * 100);

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        eyebrow="PIPELINE · RALLY · TAD"
        title="TAD Predict"
        actions={
          <>
            <span className="self-center font-mono text-xs tabular-nums text-text-muted">{selectedTotal} selected</span>
            <Button intent="primary" onClick={start} disabled={running}>
              {running ? 'Running…' : 'Start Prediction'}
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Cuts" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selectedTotal} tintClass="text-accent" />
        <StatTile label="Predicted" value={predictedTotal} tintClass="text-emerald-400" />
        <StatTile label="Pending" value={videos.length - predictedTotal} tintClass="text-text-muted" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1fr_1.6fr]">
        {/* Config */}
        <Card>
          <SectionLabel>Config</SectionLabel>
          <div className="space-y-3">
            <div>
              <label className="mb-1.5 block text-[10.5px] uppercase tracking-wide text-text-muted">Feature model</label>
              <select value={model} onChange={(e) => setModel(e.target.value as FeatureModel)} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
                {MODELS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label className="text-[10.5px] uppercase tracking-wide text-text-muted">Checkpoint</label>
                <label className="flex cursor-pointer items-center gap-1.5 text-[10.5px] text-text-muted">
                  <input type="checkbox" checked={showAllCkpts} onChange={(e) => setShowAllCkpts(e.target.checked)} className="h-3 w-3 accent-primary" />
                  show all
                </label>
              </div>
              <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
                <option value="">Select checkpoint…</option>
                {checkpoints.map((c) => (
                  <option key={c.path} value={c.path}>
                    {`${CKPT_TAG[c.kind]} ${c.name} (${c.size_mb.toFixed(1)} MB)`.trim()}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <div>
                <label className="mb-1 block text-[10px] uppercase tracking-wide text-text-muted">Threshold</label>
                <input type="number" value={threshold} min={0} max={1} step={0.05} onChange={(e) => setThreshold(Number(e.target.value))} className={cn(fieldCls, 'font-mono tabular-nums')} />
              </div>
              <div>
                <label className="mb-1 block text-[10px] uppercase tracking-wide text-text-muted">Device</label>
                <select value={device} onChange={(e) => setDevice(e.target.value as 'cuda' | 'cpu')} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
                  <option value="cuda">CUDA</option>
                  <option value="cpu">CPU</option>
                </select>
              </div>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={cutRallies} onChange={(e) => setCutRallies(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Cut rallies
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary" title="Trim each TAD rally to its serve/score boundaries using SPOT action predictions.">
              <input type="checkbox" checked={trimActions} onChange={(e) => setTrimActions(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Trim with actions
            </label>
            <Button intent="primary" onClick={start} disabled={running} className="w-full">
              Start Prediction
            </Button>
          </div>
        </Card>

        {/* Videos */}
        <Card>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5" role="tablist">
              {KIND_TABS.map((tab) => {
                const active = tab.key === kindFilter;
                return (
                  <button
                    key={tab.key}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setKindFilter(tab.key)}
                    className={cn('rounded-md px-3 py-1 font-heading text-xs transition-colors', active ? 'bg-primary text-white' : 'text-text-secondary hover:bg-white/[0.04]')}
                  >
                    {tab.label} <span className="ml-1 opacity-60">{counts[tab.key]}</span>
                  </button>
                );
              })}
            </div>
            <TriStateFilter fields={FILTER_FIELDS} value={filter} onChange={(k, next) => setFilter((f) => ({ ...f, [k]: next }))} />
          </div>

          <div className="mb-2 flex items-center justify-between text-xs text-text-muted">
            <label className="inline-flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={allVisibleSelected} onChange={(e) => setVisibleSelection(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
              Select visible
            </label>
            <span className="font-mono tabular-nums">
              {selectedTotal} / {videos.length} selected
            </span>
          </div>

          <div className="max-h-80 space-y-0.5 overflow-y-auto pr-1">
            {visible.length === 0 ? (
              <EmptyState
                icon={
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 4h18M3 12h18M3 20h18" />
                  </svg>
                }
                title={videos.length ? 'No videos match the filters' : 'No cut videos found'}
                subtitle={videos.length ? 'Adjust the kind tab or filter chips' : 'Cut some videos first'}
              />
            ) : (
              visible.map((v) => {
                const hasFeat = !!v.features?.[model];
                return (
                  <div
                    key={v.name}
                    className={cn(
                      'group flex items-center gap-3 rounded-lg border border-transparent p-2.5 transition-colors hover:border-border hover:bg-surface-50',
                      !hasFeat && 'opacity-60',
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={v.selected}
                      onChange={(e) => setVideos((prev) => prev.map((x) => (x.name === v.name ? { ...x, selected: e.target.checked } : x)))}
                      className="h-3.5 w-3.5 cursor-pointer accent-primary"
                    />
                    <span className="min-w-0 flex-1 truncate text-sm text-text-primary">{v.name}</span>
                    {v.has_annotation ? (
                      <Pill tone="emerald">ann</Pill>
                    ) : v.has_pre_annotation ? (
                      <Pill tone="sky">pre</Pill>
                    ) : null}
                    <Pill tone={hasFeat ? 'emerald' : 'muted'}>{hasFeat ? 'feat' : 'no feat'}</Pill>
                    <Pill tone={v.has_prediction ? 'primary' : 'muted'}>{v.has_prediction ? 'pred' : 'pending'}</Pill>
                  </div>
                );
              })
            )}
          </div>
        </Card>
      </div>

      {/* Progress */}
      {job && (
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <SectionLabel className="mb-0">Progress · {pct}%</SectionLabel>
            {job.status === 'failed' && (
              <Button size="sm" intent="primary" onClick={start}>
                Retry Failed
              </Button>
            )}
          </div>
          <JobProgressCard job={job} showLogs />
        </Card>
      )}
    </div>
  );
}

function Pill({ tone, children }: { tone: 'emerald' | 'sky' | 'primary' | 'muted'; children: ReactNode }) {
  const cls = {
    emerald: 'text-emerald-400 bg-emerald-500/10 ring-emerald-500/20',
    sky: 'text-sky-300 bg-sky-500/10 ring-sky-500/20',
    primary: 'text-primary-light bg-primary/15 ring-primary/25',
    muted: 'text-text-muted bg-white/5 ring-white/10',
  }[tone];
  return <span className={cn('flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ring-1', cls)}>{children}</span>;
}

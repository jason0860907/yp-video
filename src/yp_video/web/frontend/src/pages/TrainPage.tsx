import { useEffect, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { isTerminal } from '@/lib/job';
import { useSSE } from '@/lib/useSSE';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { TriStateFilter, type TriState } from '@/components/ui/TriStateFilter';
import { ProgressBar } from '@/components/job/ProgressBar';
import { JobProgress } from '@/components/job/JobProgress';
import { PerformanceChart } from '@/components/train/PerformanceChart';
import { confirm } from '@/components/feedback/confirm';
import { toast } from '@/components/feedback/toast';
import type { Job, PerfData, TrainConfigDefaults, TrainStatus, TrainVideo, VllmStatus } from '@/types/api';

type FeatureModel = 'base' | 'large' | 'giant' | 'gigantic';
type KindFilter = 'all' | 'broadcast' | 'sideline';
type FilterKey = 'annotated' | 'pre_annotated' | 'features' | 'prediction';

const MODELS: Array<{ value: FeatureModel; label: string }> = [
  { value: 'base', label: 'ViT-B (768d, 80M)' },
  { value: 'large', label: 'ViT-L (1024d, 300M)' },
  { value: 'giant', label: 'ViT-g (1408d, 1B)' },
  { value: 'gigantic', label: 'ViT-G (1664d, 2B)' },
];
const MODEL_SHORT: Record<FeatureModel, string> = { base: 'ViT-B', large: 'ViT-L', giant: 'ViT-g', gigantic: 'ViT-G' };
const KIND_TABS: Array<{ key: KindFilter; label: string }> = [
  { key: 'all', label: 'All' },
  { key: 'broadcast', label: 'Broadcast' },
  { key: 'sideline', label: 'Sideline' },
];
const FILTER_FIELDS: ReadonlyArray<{ key: FilterKey; label: string; prop: keyof TrainVideo }> = [
  { key: 'annotated', label: 'annotated', prop: 'has_annotation' },
  { key: 'pre_annotated', label: 'pre-ann', prop: 'has_pre_annotation' },
  { key: 'features', label: 'features', prop: 'has_features' },
  { key: 'prediction', label: 'predicted', prop: 'has_prediction' },
];
const EMPTY_FILTER: Record<FilterKey, TriState> = { annotated: null, pre_annotated: null, features: null, prediction: null };

const fieldCls =
  'rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';
const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const matches = (v: TrainVideo, kind: KindFilter, filter: Record<FilterKey, TriState>) => {
  if (kind !== 'all' && v.kind !== kind) return false;
  return FILTER_FIELDS.every((f) => filter[f.key] === null || !!v[f.prop] === filter[f.key]);
};

export function TrainPage() {
  const qc = useQueryClient();
  const [model, setModel] = useState<FeatureModel>('base');

  const [extractSel, setExtractSel] = useState<Set<string>>(new Set());
  const [extractKind, setExtractKind] = useState<KindFilter>('all');
  const [extractFilter, setExtractFilter] = useState(EMPTY_FILTER);
  const [featBatch, setFeatBatch] = useState(32);

  const [convSel, setConvSel] = useState<Set<string>>(new Set());
  const [convKind, setConvKind] = useState<KindFilter>('all');
  const [convFilter, setConvFilter] = useState(EMPTY_FILTER);
  const [ratio, setRatio] = useState(0.8);

  const [gpu, setGpu] = useState(0);
  const [seed, setSeed] = useState(42);
  const [advanced, setAdvanced] = useState(false);
  const [adv, setAdv] = useState({ lr: '', epochs: '', warmup: '', schedule: '', batch: '', wd: '' });
  const [balanced, setBalanced] = useState(true);
  const [alpha, setAlpha] = useState(0.5);

  const [extractJob, setExtractJob] = useState<Job | null>(null);
  const [trainJob, setTrainJob] = useState<Job | null>(null);

  const statusQuery = useQuery({
    queryKey: ['train-status', model],
    queryFn: () => apiFetch<TrainStatus>(API.train.status({ model })),
    refetchInterval: trainJob && !isTerminal(trainJob.status) ? false : 30_000,
  });
  const videosQuery = useQuery({ queryKey: ['train-videos', model], queryFn: () => apiFetch<TrainVideo[]>(API.system.videos({ model })) });
  const configQuery = useQuery({ queryKey: ['train-config'], queryFn: () => apiFetch<TrainConfigDefaults>(API.train.configDefaults) });
  const perfQuery = useQuery({ queryKey: ['train-perf', model], queryFn: () => apiFetch<PerfData>(API.train.performance({ model })) });
  const status = statusQuery.data;
  const config = configQuery.data;

  // Default selections when the video list (re)loads.
  useEffect(() => {
    const vs = videosQuery.data;
    if (!vs) return;
    setExtractSel(new Set(vs.filter((v) => !v.has_features).map((v) => v.name)));
    setConvSel(new Set(vs.filter((v) => v.has_annotation || v.has_pre_annotation).map((v) => v.name)));
  }, [videosQuery.data]);

  // Seed sampler alpha from config defaults.
  useEffect(() => {
    if (config?.sampler_alpha != null) setAlpha(config.sampler_alpha);
  }, [config?.sampler_alpha]);

  // Adopt an already-running training job.
  useEffect(() => {
    if (status?.active_train_job && !trainJob) setTrainJob(status.active_train_job);
  }, [status?.active_train_job, trainJob]);

  useSSE<Job>(extractJob && !isTerminal(extractJob.status) ? API.jobs.eventsSSE(extractJob.id) : null, (data) => {
    setExtractJob(data);
    if (isTerminal(data.status)) {
      toast[data.status === 'completed' ? 'success' : 'error'](data.status === 'completed' ? 'Features extracted!' : `Failed: ${data.error}`);
      void qc.invalidateQueries({ queryKey: ['train-status', model] });
      void qc.invalidateQueries({ queryKey: ['train-videos', model] });
    }
  });
  useSSE<Job>(trainJob && !isTerminal(trainJob.status) ? API.jobs.eventsSSE(trainJob.id) : null, (data) => {
    setTrainJob(data);
    if (data.message?.includes('mAP')) void qc.invalidateQueries({ queryKey: ['train-perf', model] });
    if (isTerminal(data.status)) {
      toast[data.status === 'completed' ? 'success' : 'error'](data.status === 'completed' ? 'Training complete!' : `Failed: ${data.error}`);
      void qc.invalidateQueries({ queryKey: ['train-status', model] });
      void qc.invalidateQueries({ queryKey: ['train-perf', model] });
    }
  });

  const videos = videosQuery.data ?? [];
  const featB = status?.features_by_model?.base ?? 0;
  const featL = status?.features_by_model?.large ?? 0;

  const extract = async () => {
    const names = videos.filter((v) => extractSel.has(v.name)).map((v) => v.name);
    if (!names.length) return toast.warning('No videos selected');
    let stopVllm = false;
    const vllm = await apiFetch<VllmStatus>(API.system.vllmStatus).catch(() => null);
    if (vllm?.status === 'running') {
      const ok = await confirm({
        title: 'Stop vLLM for feature extraction?',
        body: 'vLLM holds most of the GPU VRAM; extraction needs it and would OOM otherwise. vLLM restarts automatically once extraction finishes.',
        confirmText: 'Stop & Extract',
        variant: 'warning',
      });
      if (!ok) return;
      stopVllm = true;
    }
    try {
      const job = await apiFetch<Job>(API.train.extractFeatures, { method: 'POST', body: { videos: names, batch_size: featBatch, model, stop_vllm: stopVllm } });
      setExtractJob(job);
    } catch (e) {
      toast.error(`Failed: ${errMsg(e)}`);
    }
  };

  const convert = async () => {
    const names = videos.filter((v) => convSel.has(v.name)).map((v) => v.name);
    if (!names.length) return toast.warning('No videos selected');
    try {
      const res = await apiFetch<{ video_count: number }>(API.train.convertAnnotations, { method: 'POST', body: { train_ratio: ratio, videos: names, model } });
      toast.success(`${res.video_count} videos converted`);
    } catch (e) {
      toast.error(`Convert failed: ${errMsg(e)}`);
    }
  };

  const startTraining = async () => {
    const num = (s: string) => (s.trim() === '' ? null : (Number.isFinite(Number(s)) ? Number(s) : null));
    try {
      const job = await apiFetch<Job>(API.train.start, {
        method: 'POST',
        body: {
          gpu,
          seed,
          model,
          lr: num(adv.lr),
          epochs: num(adv.epochs),
          warmup_epochs: num(adv.warmup),
          schedule: adv.schedule || null,
          batch_size: num(adv.batch),
          weight_decay: num(adv.wd),
          balanced_sampler: balanced,
          sampler_alpha: alpha,
        },
      });
      setTrainJob(job);
    } catch (e) {
      toast.error(`Failed: ${errMsg(e)}`);
    }
  };

  const training = !!trainJob && !isTerminal(trainJob.status);

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <div className="grid grid-cols-2 gap-3.5 md:grid-cols-5">
        <StatTile label="Cuts" value={status?.cuts_count ?? 0} tintClass="text-primary-light" />
        <StatTile label="Features · B" value={featB} tintClass={featB ? 'text-emerald-400' : 'text-text-muted'} />
        <StatTile label="Features · L" value={featL} tintClass={featL ? 'text-emerald-400' : 'text-text-muted'} />
        <StatTile label="Annotations" value={status?.annotations_exist ? 'ready' : 'missing'} tintClass={status?.annotations_exist ? 'text-emerald-400' : 'text-amber-400'} />
        <StatTile label="GPU" value={status?.vllm_running ? 'shared' : 'free'} tintClass={status?.vllm_running ? 'text-amber-400' : 'text-emerald-400'} />
      </div>

      {/* Step 1 — Extract features */}
      <Card>
        <SectionLabel>1 · Extract features · V-JEPA</SectionLabel>
        <VideoList
          videos={videos}
          sel={extractSel}
          onToggle={(name, on) => setExtractSel((p) => toggle(p, name, on))}
          onToggleVisible={(on) => setExtractSel((p) => toggleVisible(p, videos, (v) => matches(v, extractKind, extractFilter), on))}
          kind={extractKind}
          setKind={setExtractKind}
          filter={extractFilter}
          setFilter={setExtractFilter}
        />
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Labeled label="Model">
            <select value={model} onChange={(e) => setModel(e.target.value as FeatureModel)} className={cn(fieldCls, 'w-48 cursor-pointer appearance-none')}>
              {MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </Labeled>
          <Labeled label="Batch">
            <input type="number" value={featBatch} min={1} max={64} onChange={(e) => setFeatBatch(Number(e.target.value))} className={cn(fieldCls, 'w-24 font-mono tabular-nums')} />
          </Labeled>
          <Button onClick={extract}>Extract</Button>
          <span className="ml-auto self-center font-mono text-xs tabular-nums text-text-muted">{extractSel.size} selected</span>
        </div>
        {extractJob && (
          <div className="mt-3 space-y-1.5">
            <ProgressBar progress={extractJob.progress} />
            <p className="text-xs text-text-muted">{extractJob.message}</p>
          </div>
        )}
      </Card>

      {/* Step 2 — Convert annotations */}
      <Card>
        <SectionLabel>2 · Convert annotations · JSONL → ActionFormer</SectionLabel>
        <VideoList
          videos={videos}
          sel={convSel}
          onToggle={(name, on) => setConvSel((p) => toggle(p, name, on))}
          onToggleVisible={(on) => setConvSel((p) => toggleVisible(p, videos, (v) => matches(v, convKind, convFilter), on))}
          kind={convKind}
          setKind={setConvKind}
          filter={convFilter}
          setFilter={setConvFilter}
        />
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Labeled label="Train ratio">
            <input type="number" value={ratio} min={0.1} max={0.99} step={0.05} onChange={(e) => setRatio(Number(e.target.value))} className={cn(fieldCls, 'w-28 font-mono tabular-nums')} />
          </Labeled>
          <Button onClick={convert}>Convert</Button>
          <span className="ml-auto self-center font-mono text-xs tabular-nums text-text-muted">{convSel.size} selected</span>
        </div>
      </Card>

      {/* Step 3 — Train */}
      <Card>
        <SectionLabel>3 · Train model · ActionFormer</SectionLabel>
        <div className="flex flex-wrap items-end gap-3">
          <Labeled label="GPU">
            <input type="number" value={gpu} min={0} max={7} onChange={(e) => setGpu(Number(e.target.value))} className={cn(fieldCls, 'w-20 font-mono tabular-nums')} />
          </Labeled>
          <Labeled label="Seed">
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className={cn(fieldCls, 'w-24 font-mono tabular-nums')} />
          </Labeled>
          <span className="self-center text-[11px] text-text-muted">
            Features: <span className="font-medium text-text-primary">{MODEL_SHORT[model]}</span>
          </span>
          <Button size="sm" onClick={() => setAdvanced((o) => !o)}>
            Advanced {advanced ? '▴' : '▾'}
          </Button>
          <Button intent="primary" onClick={startTraining} disabled={training}>
            {training ? 'Training…' : 'Start Training'}
          </Button>
        </div>

        {advanced && (
          <div className="mt-3 rounded-xl border border-border bg-surface-50 p-4">
            <div className="mb-3 text-[11px] uppercase tracking-wider text-text-muted">Optimizer / schedule — blank = config default</div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              <AdvField label="Learning rate" value={adv.lr} placeholder={config?.lr} onChange={(v) => setAdv((a) => ({ ...a, lr: v }))} />
              <AdvField label="Epochs (after warmup)" value={adv.epochs} placeholder={config?.epochs} onChange={(v) => setAdv((a) => ({ ...a, epochs: v }))} />
              <AdvField label="Warmup epochs" value={adv.warmup} placeholder={config?.warmup_epochs} onChange={(v) => setAdv((a) => ({ ...a, warmup: v }))} />
              <Labeled label="Schedule">
                <select value={adv.schedule} onChange={(e) => setAdv((a) => ({ ...a, schedule: e.target.value }))} className={cn(fieldCls, 'w-full cursor-pointer appearance-none')}>
                  <option value="">(config default{config?.schedule ? `: ${config.schedule}` : ''})</option>
                  <option value="cosine">cosine</option>
                  <option value="constant">constant</option>
                  <option value="multistep">multistep</option>
                </select>
              </Labeled>
              <AdvField label="Batch size" value={adv.batch} placeholder={config?.batch_size} onChange={(v) => setAdv((a) => ({ ...a, batch: v }))} />
              <AdvField label="Weight decay" value={adv.wd} placeholder={config?.weight_decay} onChange={(v) => setAdv((a) => ({ ...a, wd: v }))} />
            </div>
            <div className="mt-3 grid grid-cols-1 items-end gap-3 md:grid-cols-3">
              <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
                <input type="checkbox" checked={balanced} onChange={(e) => setBalanced(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
                Balanced sampler (per source)
              </label>
              <Labeled label="Sampler alpha (0=uniform, 1=balanced)">
                <input type="number" value={alpha} min={0} max={1} step={0.1} onChange={(e) => setAlpha(Number(e.target.value))} className={cn(fieldCls, 'w-full font-mono tabular-nums')} />
              </Labeled>
            </div>
          </div>
        )}

        {trainJob && (
          <div className="mt-3">
            <JobProgress job={trainJob} detail="" showLogs truncateMsg={false} />
          </div>
        )}
      </Card>

      {perfQuery.data && <PerformanceChart data={perfQuery.data} />}
    </div>
  );
}

// ── helpers ──
const toggle = (set: Set<string>, name: string, on: boolean) => {
  const next = new Set(set);
  if (on) next.add(name);
  else next.delete(name);
  return next;
};
const toggleVisible = (set: Set<string>, videos: TrainVideo[], pred: (v: TrainVideo) => boolean, on: boolean) => {
  const next = new Set(set);
  videos.filter(pred).forEach((v) => (on ? next.add(v.name) : next.delete(v.name)));
  return next;
};

function Labeled({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block space-y-1">
      <span className="block text-[10.5px] uppercase tracking-wide text-text-muted">{label}</span>
      {children}
    </label>
  );
}

function AdvField({ label, value, placeholder, onChange }: { label: string; value: string; placeholder?: number; onChange: (v: string) => void }) {
  return (
    <Labeled label={label}>
      <input type="number" step="any" value={value} placeholder={placeholder != null ? String(placeholder) : ''} onChange={(e) => onChange(e.target.value)} className={cn(fieldCls, 'w-full font-mono tabular-nums')} />
    </Labeled>
  );
}

interface VideoListProps {
  videos: TrainVideo[];
  sel: Set<string>;
  onToggle: (name: string, on: boolean) => void;
  onToggleVisible: (on: boolean) => void;
  kind: KindFilter;
  setKind: (k: KindFilter) => void;
  filter: Record<FilterKey, TriState>;
  setFilter: (f: Record<FilterKey, TriState>) => void;
}

function VideoList({ videos, sel, onToggle, onToggleVisible, kind, setKind, filter, setFilter }: VideoListProps) {
  const visible = videos.filter((v) => matches(v, kind, filter));
  const counts = { all: videos.length, broadcast: videos.filter((v) => v.kind === 'broadcast').length, sideline: videos.filter((v) => v.kind === 'sideline').length };
  const allSel = visible.length > 0 && visible.every((v) => sel.has(v.name));

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
          {KIND_TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setKind(t.key)}
              className={cn('rounded-md px-3 py-1 font-heading text-xs transition-colors', t.key === kind ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]')}
            >
              {t.label} <span className="ml-1 opacity-60">{counts[t.key]}</span>
            </button>
          ))}
        </div>
        <TriStateFilter fields={FILTER_FIELDS} value={filter} onChange={(k, next) => setFilter({ ...filter, [k]: next })} />
      </div>
      <div className="mb-2 flex items-center justify-between text-xs text-text-muted">
        <label className="inline-flex cursor-pointer items-center gap-2">
          <input type="checkbox" checked={allSel} onChange={(e) => onToggleVisible(e.target.checked)} className="h-3.5 w-3.5 accent-primary" />
          Select visible
        </label>
        <span className="font-mono tabular-nums">
          {visible.filter((v) => sel.has(v.name)).length} / {visible.length}
        </span>
      </div>
      <div className="max-h-72 space-y-0.5 overflow-auto pr-1">
        {visible.length === 0 ? (
          <EmptyState
            icon={
              <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 4h18M3 12h18M3 20h18" />
              </svg>
            }
            title={videos.length ? 'No videos match the filter' : 'No cut videos found'}
          />
        ) : (
          visible.map((v) => (
            <label key={v.name} className="group flex w-max min-w-full items-center gap-3 rounded-lg border border-transparent p-2.5 transition-colors hover:border-border hover:bg-surface-50">
              <input type="checkbox" checked={sel.has(v.name)} onChange={(e) => onToggle(v.name, e.target.checked)} className="h-3.5 w-3.5 flex-shrink-0 accent-primary" />
              <span className="flex-1 whitespace-nowrap text-sm text-text-primary">{v.name}</span>
              {v.has_annotation ? <Pill tone="emerald">ann</Pill> : v.has_pre_annotation ? <Pill tone="sky">pre</Pill> : null}
              {v.has_features && <Pill tone="emerald">feat</Pill>}
              {v.has_prediction && <Pill tone="primary">pred</Pill>}
            </label>
          ))
        )}
      </div>
    </div>
  );
}

function Pill({ tone, children }: { tone: 'emerald' | 'sky' | 'primary'; children: ReactNode }) {
  const cls = {
    emerald: 'text-emerald-400 bg-emerald-500/10 ring-emerald-500/20',
    sky: 'text-sky-300 bg-sky-500/10 ring-sky-500/20',
    primary: 'text-primary-light bg-primary/15 ring-primary/25',
  }[tone];
  return <span className={cn('flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ring-1', cls)}>{children}</span>;
}

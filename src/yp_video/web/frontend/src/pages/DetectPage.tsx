import { useEffect, useState } from 'react';
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
import { toast } from '@/components/feedback/toast';
import type { CutKind, Job, VideoMeta, VllmStatus } from '@/types/api';

type KindFilter = 'all' | CutKind;
type DetectVideo = VideoMeta & { selected: boolean };

const KIND_TABS: Array<{ key: KindFilter; label: string }> = [
  { key: 'all', label: 'All' },
  { key: 'broadcast', label: 'Broadcast' },
  { key: 'sideline', label: 'Sideline' },
];

interface Settings {
  batch_size: number;
  clip_duration: number;
  slide_interval: number;
  min_duration: number;
  min_score: number;
}
const DEFAULTS: Settings = { batch_size: 16, clip_duration: 6, slide_interval: 2, min_duration: 3, min_score: 0.5 };

const SETTING_FIELDS: Array<{ key: keyof Settings; label: string; min: number; max?: number; step: number }> = [
  { key: 'batch_size', label: 'Batch size', min: 1, max: 128, step: 1 },
  { key: 'clip_duration', label: 'Clip duration (s)', min: 1, step: 0.5 },
  { key: 'slide_interval', label: 'Slide interval (s)', min: 0.5, step: 0.5 },
  { key: 'min_duration', label: 'Min duration (s)', min: 0, step: 0.5 },
  { key: 'min_score', label: 'Min score', min: 0, max: 1, step: 0.1 },
];

const fieldCls =
  'w-full rounded-lg border border-border-light bg-surface-50 px-3 py-2.5 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';
const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const isTerminal = (s: Job['status']) => s === 'completed' || s === 'failed' || s === 'cancelled';

export function DetectPage() {
  const qc = useQueryClient();
  const [videos, setVideos] = useState<DetectVideo[]>([]);
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [job, setJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);
  const [batchTouched, setBatchTouched] = useState(false);

  const videosQuery = useQuery({
    queryKey: ['system-videos'],
    queryFn: () => apiFetch<VideoMeta[]>(API.system.videos()),
  });
  const vllmQuery = useQuery({
    queryKey: ['vllm-status'],
    queryFn: () => apiFetch<VllmStatus>(API.system.vllmStatus),
  });

  useEffect(() => {
    if (videosQuery.data) setVideos(videosQuery.data.map((v) => ({ ...v, selected: !v.has_detection })));
  }, [videosQuery.data]);

  useEffect(() => {
    const seqs = vllmQuery.data?.max_num_seqs;
    if (seqs && !batchTouched) setSettings((s) => ({ ...s, batch_size: seqs }));
  }, [vllmQuery.data?.max_num_seqs, batchTouched]);

  useSSE<Job>(running && job ? API.jobs.eventsSSE(job.id) : null, (data) => {
    setJob(data);
    if (isTerminal(data.status)) {
      setRunning(false);
      if (data.status === 'failed') toast.error(`Detection failed: ${data.error || 'Unknown error'}`);
      else toast.success(data.message || 'Detection complete!');
      void qc.invalidateQueries({ queryKey: ['system-videos'] });
    }
  });

  const matchesFilter = (v: DetectVideo) => kindFilter === 'all' || v.kind === kindFilter;
  const visible = videos.filter(matchesFilter);
  const counts = {
    all: videos.length,
    broadcast: videos.filter((v) => v.kind === 'broadcast').length,
    sideline: videos.filter((v) => v.kind === 'sideline').length,
  };
  const selectedTotal = videos.filter((v) => v.selected).length;
  const detectedTotal = videos.filter((v) => v.has_detection).length;
  const selectedVisible = visible.filter((v) => v.selected).length;

  const setVisibleSelection = (fn: (v: DetectVideo) => boolean) =>
    setVideos((prev) => prev.map((v) => (matchesFilter(v) ? { ...v, selected: fn(v) } : v)));

  const startDetection = async () => {
    const selected = videos.filter((v) => v.selected).map((v) => v.name);
    if (selected.length === 0) {
      toast.warning('No videos selected');
      return;
    }
    setRunning(true);
    try {
      const started = await apiFetch<Job>(API.detect.start, { method: 'POST', body: { videos: selected, ...settings } });
      setJob(started);
    } catch (e) {
      setRunning(false);
      toast.error(`Failed to start detection: ${errMsg(e)}`);
    }
  };

  const pct = Math.round((job?.progress ?? 0) * 100);

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        eyebrow="PIPELINE · RALLY · TAD"
        title="Rally Predict"
        actions={
          <>
            <span className="self-center font-mono text-xs tabular-nums text-text-muted">{selectedTotal} selected</span>
            <Button intent="primary" onClick={startDetection} disabled={running}>
              {running ? 'Running…' : 'Run detection'}
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Cuts" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selectedTotal} tintClass="text-accent" />
        <StatTile label="Detected" value={detectedTotal} tintClass="text-emerald-400" />
        <StatTile label="Pending" value={videos.length - detectedTotal} tintClass="text-text-muted" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1fr_1.6fr]">
        {/* Config */}
        <Card>
          <SectionLabel>Config</SectionLabel>
          <div className="space-y-3">
            {SETTING_FIELDS.map((f) => (
              <div key={f.key}>
                <label className="mb-1.5 block text-[10.5px] uppercase tracking-wide text-text-muted">{f.label}</label>
                <input
                  type="number"
                  value={settings[f.key]}
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  onChange={(e) => {
                    if (f.key === 'batch_size') setBatchTouched(true);
                    setSettings((s) => ({ ...s, [f.key]: Number(e.target.value) }));
                  }}
                  className={cn(fieldCls, 'font-mono tabular-nums')}
                />
              </div>
            ))}
            <Button intent="primary" onClick={startDetection} disabled={running} className="w-full">
              Run detection
            </Button>
          </div>
        </Card>

        {/* Cut videos */}
        <Card>
          <div className="mb-3 flex items-center justify-between gap-3">
            <SectionLabel className="mb-0">Cut videos</SectionLabel>
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={() => setVisibleSelection(() => true)}>
                Select All
              </Button>
              <Button size="sm" onClick={() => setVisibleSelection(() => false)}>
                Deselect All
              </Button>
              <Button size="sm" intent="primary" onClick={() => setVisibleSelection((v) => !v.has_detection)}>
                Undetected
              </Button>
            </div>
          </div>

          <div className="mb-3 inline-flex rounded-lg border border-border bg-surface-50 p-0.5" role="tablist">
            {KIND_TABS.map((tab) => {
              const active = tab.key === kindFilter;
              return (
                <button
                  key={tab.key}
                  type="button"
                  aria-pressed={active}
                  onClick={() => setKindFilter(tab.key)}
                  className={cn(
                    'rounded-md px-3 py-1 font-heading text-xs transition-colors duration-150',
                    active ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]',
                  )}
                >
                  {tab.label} <span className="ml-1 opacity-60">{counts[tab.key]}</span>
                </button>
              );
            })}
            <span className="self-center px-2 font-mono text-[11px] tabular-nums text-text-muted">
              {selectedVisible}/{visible.length}
            </span>
          </div>

          <div className="max-h-80 space-y-0.5 overflow-y-auto pr-1">
            {visible.length === 0 ? (
              <EmptyState
                icon={
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                }
                title={videos.length ? `No ${kindFilter} cuts` : 'No cut videos found'}
                subtitle={videos.length ? 'Switch tab or cut more videos' : 'Cut some videos first'}
              />
            ) : (
              visible.map((v) => (
                <div
                  key={v.name}
                  className="group flex items-center gap-3 rounded-lg border border-transparent p-2.5 transition-colors hover:border-border hover:bg-surface-50"
                >
                  <input
                    type="checkbox"
                    checked={v.selected}
                    onChange={(e) =>
                      setVideos((prev) => prev.map((x) => (x.name === v.name ? { ...x, selected: e.target.checked } : x)))
                    }
                    className="h-3.5 w-3.5 flex-shrink-0 cursor-pointer accent-primary"
                  />
                  <KindBadge kind={v.kind} />
                  <span className="min-w-0 flex-1 truncate text-sm text-text-primary">{v.name}</span>
                  {v.has_detection ? (
                    <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-400 ring-1 ring-emerald-500/20">
                      <span className="h-1.5 w-1.5 rounded-full bg-current" />
                      detected
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5 rounded-full bg-ink/5 px-2.5 py-0.5 text-[11px] font-medium text-text-muted ring-1 ring-ink/10">
                      <span className="h-1.5 w-1.5 rounded-full bg-current" />
                      pending
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* Inference progress */}
      {job && (
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <SectionLabel className="mb-0">Inference progress</SectionLabel>
            <span className={cn('flex items-center gap-1.5 font-mono text-[11px] uppercase', running ? 'text-primary-light' : job.status === 'failed' ? 'text-red-400' : 'text-emerald-400')}>
              <span className={cn('h-1.5 w-1.5 rounded-full bg-current', running && 'animate-pulse-dot')} />
              {job.status}
            </span>
          </div>
          <div className="mb-2.5 flex items-baseline gap-3">
            <span className="font-mono text-[30px] font-bold tabular-nums text-text-primary">{pct}%</span>
            {job.message && <span className="text-sm text-text-muted">{job.message}</span>}
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-ink/[0.06]">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${pct}%`, background: 'linear-gradient(90deg, rgb(var(--primary)), #FB923C)' }}
            />
          </div>
          {job.error && <p className="mt-3 text-[11px] text-red-400/80">{job.error}</p>}
          {job.status === 'failed' && Array.isArray(job.logs) && job.logs.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-[10px] text-text-muted hover:text-text-primary">
                Show logs ({job.logs.length} lines)
              </summary>
              <pre className="mt-1 max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-ink/5 bg-black/40 p-2 font-mono text-[10px] text-red-300/80">
                {job.logs.join('\n')}
              </pre>
            </details>
          )}
          {job.status === 'failed' && (
            <div className="mt-3">
              <Button size="sm" intent="primary" onClick={startDetection}>
                Retry Failed
              </Button>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function KindBadge({ kind }: { kind: CutKind }) {
  return kind === 'sideline' ? (
    <span className="flex-shrink-0 rounded bg-amber-500/10 px-1.5 py-0.5 font-heading text-[10px] uppercase tracking-wide text-amber-300 ring-1 ring-amber-500/20">
      side
    </span>
  ) : (
    <span className="flex-shrink-0 rounded bg-sky-500/10 px-1.5 py-0.5 font-heading text-[10px] uppercase tracking-wide text-sky-300 ring-1 ring-sky-500/20">
      cast
    </span>
  );
}

import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { useSSE } from '@/lib/useSSE';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { JobProgress } from '@/components/job/JobProgress';
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
  { key: 'batch_size', label: 'Batch Size', min: 1, max: 128, step: 1 },
  { key: 'clip_duration', label: 'Clip Duration', min: 1, step: 0.5 },
  { key: 'slide_interval', label: 'Slide Interval', min: 0.5, step: 0.5 },
  { key: 'min_duration', label: 'Min Duration (s)', min: 0, step: 0.5 },
  { key: 'min_score', label: 'Min Score', min: 0, max: 1, step: 0.1 },
];

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

  // Server data drives the list; selection defaults to undetected on each load
  // (matches reloading after a detection run).
  useEffect(() => {
    if (videosQuery.data) {
      setVideos(videosQuery.data.map((v) => ({ ...v, selected: !v.has_detection })));
    }
  }, [videosQuery.data]);

  // Seed batch size from the server's max_num_seqs until the user edits it.
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
      const started = await apiFetch<Job>(API.detect.start, {
        method: 'POST',
        body: { videos: selected, ...settings },
      });
      setJob(started);
    } catch (e) {
      setRunning(false);
      toast.error(`Failed to start detection: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-6">
      <PageHeader title="Rally Predict" subtitle="Run rally detection on cut videos" />

      {/* Video picker */}
      <Card>
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="font-heading text-sm font-semibold text-text-primary">Videos</h3>
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

        <div className="mb-3 inline-flex rounded-lg border border-border bg-surface-100 p-0.5" role="tablist">
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
                  active ? 'bg-primary text-white' : 'text-text-secondary hover:bg-white/[0.04]',
                )}
              >
                {tab.label} <span className="ml-1 opacity-60">{counts[tab.key]}</span>
              </button>
            );
          })}
        </div>

        <div className="max-h-72 space-y-0.5 overflow-y-auto pr-1">
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
                className="group flex items-center gap-3 rounded-xl border border-transparent p-2.5 transition-all duration-200 hover:border-white/5 hover:bg-white/[0.03]"
              >
                <input
                  type="checkbox"
                  checked={v.selected}
                  onChange={(e) =>
                    setVideos((prev) => prev.map((x) => (x.name === v.name ? { ...x, selected: e.target.checked } : x)))
                  }
                  className="h-3.5 w-3.5 cursor-pointer accent-primary"
                />
                <KindBadge kind={v.kind} />
                <span className="flex-1 truncate text-sm text-text-primary transition-colors group-hover:text-white">
                  {v.name}
                </span>
                {v.has_detection ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-400 ring-1 ring-emerald-500/20">
                    <span className="h-1.5 w-1.5 rounded-full bg-current" />
                    detected
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-white/5 px-2.5 py-0.5 text-[11px] font-medium text-text-muted ring-1 ring-white/10">
                    <span className="h-1.5 w-1.5 rounded-full bg-current" />
                    pending
                  </span>
                )}
              </div>
            ))
          )}
        </div>
      </Card>

      {/* Settings */}
      <Card>
        <h3 className="font-heading text-sm font-semibold text-text-primary">Detection Settings</h3>
        <p className="mt-1 text-sm text-text-muted">VLM sliding-window + rally conversion parameters</p>
        <div className="mt-5 grid grid-cols-3 gap-4">
          {SETTING_FIELDS.map((f) => (
            <div key={f.key}>
              <label className="mb-1.5 block text-[11px] uppercase tracking-wider text-text-muted">{f.label}</label>
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
                className="w-full rounded-xl border border-border-light bg-surface-100 px-3.5 py-2.5 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15"
              />
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <Button intent="primary" onClick={startDetection} disabled={running}>
            Start Detection
          </Button>
          <span className="ml-auto font-mono text-xs tabular-nums text-text-muted">
            {visible.length ? `${selectedVisible} / ${visible.length} selected` : ''}
          </span>
        </div>
      </Card>

      {/* Progress */}
      {job && (
        <Card>
          <div className="mb-3 flex items-center gap-2.5">
            <span className={cn('h-2 w-2 rounded-full bg-primary-light', running && 'animate-pulse-dot')} />
            <h3 className="font-heading text-sm font-semibold text-text-primary">Progress</h3>
          </div>
          <JobProgress job={job} showLogs />
          {job.status === 'failed' && (
            <div className="pt-3">
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
    <span className="inline-flex items-center rounded bg-amber-500/10 px-1.5 py-0.5 font-heading text-[10px] uppercase tracking-wide text-amber-300 ring-1 ring-amber-500/20">
      side
    </span>
  ) : (
    <span className="inline-flex items-center rounded bg-sky-500/10 px-1.5 py-0.5 font-heading text-[10px] uppercase tracking-wide text-sky-300 ring-1 ring-sky-500/20">
      cast
    </span>
  );
}

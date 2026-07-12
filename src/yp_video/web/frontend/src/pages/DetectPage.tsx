import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { LiveJob } from '@/components/job/LiveJob';
import { toast } from '@/components/feedback/toast';
import type { Job, VideoMeta, VllmStatus } from '@/types/api';

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

export function DetectPage() {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [job, setJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);
  const [batchTouched, setBatchTouched] = useState(false);
  const seeded = useRef(false);

  const videosQuery = useQuery({
    queryKey: ['system-videos'],
    queryFn: () => apiFetch<VideoMeta[]>(API.system.videos),
  });
  const vllmQuery = useQuery({
    queryKey: ['vllm-status'],
    queryFn: () => apiFetch<VllmStatus>(API.system.vllmStatus),
  });

  const videos = videosQuery.data ?? [];

  // Preselect undetected videos on first load only — later refetches must not
  // clobber whatever the user has picked in the meantime.
  useEffect(() => {
    if (videosQuery.data && !seeded.current) {
      seeded.current = true;
      setSelected(new Set(videosQuery.data.filter((v) => !v.has_detection).map((v) => v.name)));
    }
  }, [videosQuery.data]);

  useEffect(() => {
    const seqs = vllmQuery.data?.max_num_seqs;
    if (seqs && !batchTouched) setSettings((s) => ({ ...s, batch_size: seqs }));
  }, [vllmQuery.data?.max_num_seqs, batchTouched]);

  // Query invalidation happens inside LiveJob; this only handles page state.
  const onJobSettled = (data: Job) => {
    setRunning(false);
    if (data.status === 'failed') {
      toast.error(`Detection failed: ${data.error || 'Unknown error'}`);
    } else {
      toast.success(data.message || 'Detection complete!');
      setSelected(new Set()); // batch done — keeping it selected invites an accidental rerun
    }
  };

  const detectedTotal = videos.filter((v) => v.has_detection).length;

  const startDetection = async () => {
    const names = [...selected];
    if (names.length === 0) {
      toast.warning('No videos selected');
      return;
    }
    setRunning(true);
    try {
      const started = await apiFetch<Job>(API.detect.start, { method: 'POST', body: { videos: names, ...settings } });
      setJob(started);
    } catch (e) {
      setRunning(false);
      toast.error(`Failed to start detection: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <>
            <span className="self-center font-mono text-xs tabular-nums text-text-muted">{selected.size} selected</span>
            <Button intent="primary" onClick={startDetection} disabled={running}>
              {running ? 'Running…' : 'Run detection'}
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Cuts" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Detected" value={detectedTotal} tintClass="text-primary-light" />
        <StatTile label="Pending" value={videos.length - detectedTotal} tintClass="text-text-muted" />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[20rem_minmax(0,1fr)]">
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
          <VideoMultiSelectList
            title="Cut videos"
            videos={videos}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              { value: 'all', label: 'All', predicate: () => true },
              { value: 'pending', label: 'Pending', predicate: (v) => !v.has_detection },
              { value: 'detected', label: 'Detected', predicate: (v) => Boolean(v.has_detection) },
            ]}
            quickSelects={[{ label: 'Undetected', predicate: (v) => !v.has_detection }]}
            renderMeta={(v) => (v.has_detection ? <Badge tone="success">detected</Badge> : <Badge tone="neutral">pending</Badge>)}
            maxHeightClass="max-h-80"
            emptyTitle="No cut videos found"
            emptySubtitle="Cut some videos first"
          />
        </Card>
      </div>

      {/* Same shared job card as every other page's job list */}
      {job && (
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <SectionLabel className="!mb-0">Rally Predict jobs</SectionLabel>
            {job.status === 'failed' && (
              <Button size="sm" intent="primary" onClick={startDetection}>
                Retry Failed
              </Button>
            )}
          </div>
          <LiveJob job={job} onUpdate={setJob} onSettled={onJobSettled} />
        </Card>
      )}
    </div>
  );
}

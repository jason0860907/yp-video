import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { formatTimePrecise } from '@/lib/format';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { CutKind } from '@/types/api';

interface Segment {
  name: string;
  start: number;
  end: number;
  auto: boolean;
}

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const stemOf = (name: string) => name.replace(/\.[^.]+$/, '');

export function CutPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [video, setVideo] = useState('');
  const [segments, setSegments] = useState<Segment[]>([]);
  const [markStart, setMarkStart] = useState<number | null>(null);
  const [kind, setKind] = useState<CutKind>('broadcast');
  const [time, setTime] = useState(0);
  const [exporting, setExporting] = useState(false);

  const videosQuery = useQuery({ queryKey: ['cut-videos'], queryFn: () => apiFetch<string[]>(API.cut.videos) });
  const videos = videosQuery.data ?? [];

  const renumber = (segs: Segment[]): Segment[] => {
    const stem = stemOf(video);
    return segs.map((s, i) => (s.auto ? { ...s, name: segs.length === 1 ? stem : `${stem}_set${i + 1}` } : s));
  };

  const doMarkStart = () => {
    const el = videoRef.current;
    if (!el || !el.src) return;
    setMarkStart(el.currentTime);
  };
  const doMarkEnd = () => {
    const el = videoRef.current;
    if (markStart == null || !el || !el.src) return;
    const end = el.currentTime;
    if (end <= markStart) {
      toast.warning('End must be after start');
      return;
    }
    setSegments((prev) => renumber([...prev, { name: '', start: markStart, end, auto: true }]));
    setMarkStart(null);
  };

  // Keyboard transport: space play/pause, [ ] mark, arrows skip 5s.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      const el = videoRef.current;
      if (!el) return;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          el.paused ? void el.play() : el.pause();
          break;
        case '[':
          e.preventDefault();
          doMarkStart();
          break;
        case ']':
          e.preventDefault();
          doMarkEnd();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          el.currentTime = Math.max(0, el.currentTime - 5);
          break;
        case 'ArrowRight':
          e.preventDefault();
          el.currentTime += 5;
          break;
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  const selectVideo = (name: string) => {
    setVideo(name);
    setSegments([]);
    setMarkStart(null);
    const el = videoRef.current;
    if (el && name) {
      el.src = apiUrl(API.cut.video(name));
      el.load();
    }
  };

  const deleteSource = async (context = '') => {
    if (!video) return false;
    const ok = await confirm({
      title: 'Delete source video?',
      body: `${context ? context + '\n\n' : ''}This permanently removes the raw file from local storage:\n${video}\n\nExported cut segments are not affected.`,
      confirmText: 'Delete',
      cancelText: 'Keep',
      variant: 'danger',
    });
    if (!ok) return false;
    try {
      await apiFetch(API.cut.video(video), { method: 'DELETE' });
      toast.success(`Deleted ${video}`);
      const el = videoRef.current;
      el?.pause();
      el?.removeAttribute('src');
      el?.load();
      setVideo('');
      setSegments([]);
      setMarkStart(null);
      void videosQuery.refetch();
      return true;
    } catch (e) {
      toast.error(`Delete failed: ${errMsg(e)}`);
      return false;
    }
  };

  const exportAll = async () => {
    if (segments.length === 0) return toast.warning('No segments to export');
    if (!video) return toast.warning('No video selected');
    setExporting(true);
    try {
      const res = await apiFetch<{ success: string[]; failed: string[] }>(API.cut.export, {
        method: 'POST',
        body: { source: video, segments, kind },
      });
      const failed = res.failed?.length ?? 0;
      if (failed) toast.warning(`Exported ${res.success.length} segments, ${failed} failed`);
      else toast.success(`Exported ${res.success.length} segments`);
      if (failed === 0 && res.success.length > 0) {
        await deleteSource(`All ${res.success.length} segments exported successfully.`);
      }
    } catch (e) {
      toast.error(`Export failed: ${errMsg(e)}`);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        eyebrow="PIPELINE · INGEST"
        title="Cut into sets"
        actions={
          <>
            <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
              {(['broadcast', 'sideline'] as CutKind[]).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setKind(k)}
                  className={cn('rounded-md px-2.5 py-1 font-heading text-xs capitalize transition-colors', kind === k ? 'bg-primary text-white' : 'text-text-secondary hover:bg-white/[0.04]')}
                >
                  {k}
                </button>
              ))}
            </div>
            <Button intent="primary" onClick={exportAll} disabled={exporting}>
              {exporting ? 'Exporting…' : 'Export All'}
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1.5fr_1fr]">
        {/* Player */}
        <Card>
          <div className="mb-3 flex items-center gap-3">
            <label className="flex-shrink-0 text-sm font-medium text-text-secondary">Video</label>
            <select
              value={video}
              onChange={(e) => selectVideo(e.target.value)}
              className="flex-1 cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15"
            >
              <option value="">Select a video…</option>
              {videos.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
          <div className="overflow-hidden rounded-2xl bg-black shadow-lg shadow-black/40 ring-1 ring-white/[0.06]">
            <video ref={videoRef} className="max-h-[50vh] w-full" controls onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)} />
          </div>
          <div className="mt-3 flex items-center justify-between">
            <span className="rounded-lg border border-border bg-surface-50 px-3 py-1.5 font-mono text-sm tabular-nums text-text-primary">
              {formatTimePrecise(time)}
            </span>
            <div className="flex items-center gap-2">
              <Button size="sm" intent="primary" onClick={doMarkStart}>
                Mark Start [
              </Button>
              <Button size="sm" intent="primary" onClick={doMarkEnd}>
                Mark End ]
              </Button>
            </div>
          </div>
          {markStart != null && (
            <div className="mt-3 flex items-center gap-3 rounded-xl border border-primary/20 bg-primary/10 p-3">
              <span className="h-1.5 w-1.5 rounded-full bg-primary-light animate-pulse-dot" />
              <span className="text-xs text-primary-light">
                Start at <strong className="font-mono">{formatTimePrecise(markStart)}</strong> — press ] to set end
              </span>
            </div>
          )}
        </Card>

        {/* Segments */}
        <Card>
          <SectionLabel>Segments · {segments.length}</SectionLabel>
          {segments.length === 0 ? (
            <EmptyState
              icon={
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M14.121 14.121A3 3 0 109.879 9.879m4.242 4.242L9.879 9.879" />
                </svg>
              }
              title="No segments yet"
              subtitle="Use [ and ] to mark start / end points"
            />
          ) : (
            <div className="space-y-1.5">
              {segments.map((s, i) => (
                <div key={i} className="group flex items-center gap-3 rounded-xl border border-border bg-surface-50 px-3 py-2.5">
                  <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md border border-border bg-surface-200 font-heading text-[10px] text-text-muted">
                    {i + 1}
                  </span>
                  <input
                    type="text"
                    value={s.name}
                    onChange={(e) => setSegments((prev) => prev.map((x, j) => (j === i ? { ...x, name: e.target.value, auto: false } : x)))}
                    className="min-w-0 flex-1 border-0 border-b border-transparent bg-transparent p-0 font-heading text-sm text-text-primary focus:border-primary-light focus:outline-none focus:ring-0"
                  />
                  <span className="font-mono text-xs tabular-nums text-text-muted">{(s.end - s.start).toFixed(1)}s</span>
                  <div className="ml-1 flex items-center gap-1.5 opacity-0 transition-opacity group-hover:opacity-100">
                    <button
                      type="button"
                      onClick={() => {
                        const el = videoRef.current;
                        if (el) {
                          el.currentTime = s.start;
                          void el.play();
                        }
                      }}
                      className="text-xs text-primary-light hover:underline"
                    >
                      Preview
                    </button>
                    <button type="button" onClick={() => setSegments((prev) => renumber(prev.filter((_, j) => j !== i)))} className="text-xs text-red-400 hover:underline">
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Delete source */}
      <Card>
        <div className="flex items-center justify-between gap-4">
          <div className="min-w-0">
            <p className="font-heading text-sm font-medium text-text-primary">Delete source video</p>
            <p className="mt-0.5 text-[11px] text-text-muted">Permanently removes the raw file from local storage. Cut segments are kept.</p>
          </div>
          <Button intent="danger" onClick={() => deleteSource()} disabled={!video}>
            Delete Video
          </Button>
        </div>
      </Card>

      <p className="px-1 text-[11px] text-text-muted">
        <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">[ ]</kbd> mark ·{' '}
        <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">← →</kbd> skip 5s ·{' '}
        <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">space</kbd> play
      </p>
    </div>
  );
}

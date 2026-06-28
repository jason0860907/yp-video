import { useMemo, useState } from 'react';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { formatBytes, formatDuration, formatSpeed } from '@/lib/format';
import { useSSE } from '@/lib/useSSE';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { StatusBadge } from '@/components/job/StatusBadge';
import { ProgressBar } from '@/components/job/ProgressBar';
import { toast } from '@/components/feedback/toast';

interface PlaylistResponse {
  title: string;
  videos: Array<{ id: string; title: string; duration?: number; url: string }>;
}
interface VideoProgress {
  video_id?: string;
  status?: string;
  percent?: number;
  downloaded?: number;
  total?: number;
  speed?: number;
  eta?: number;
}
interface DownloadVideo {
  id: string;
  title: string;
  duration?: number;
  url: string;
  selected: boolean;
  status: string;
  progress: VideoProgress | null;
}

const QUALITIES = [
  { value: 'best', label: 'Best' },
  { value: '1080', label: '1080p' },
  { value: '720', label: '720p' },
  { value: '480', label: '480p' },
];

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

export function DownloadPage() {
  const [url, setUrl] = useState('');
  const [fetching, setFetching] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [title, setTitle] = useState<string | null>(null);
  const [videos, setVideos] = useState<DownloadVideo[]>([]);
  const [quality, setQuality] = useState('best');
  const [downloading, setDownloading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Live per-video progress while a session is active.
  useSSE<VideoProgress>(downloading && sessionId ? API.download.progressSSE(sessionId) : null, (data) => {
    if (data.video_id) {
      setVideos((prev) =>
        prev.map((v) => (v.id === data.video_id ? { ...v, status: data.status ?? v.status, progress: data } : v)),
      );
    }
    if (data.status === 'complete') {
      setDownloading(false);
      toast.success('Download complete!');
    }
  });

  const selectedCount = useMemo(() => videos.filter((v) => v.selected).length, [videos]);

  const fetchPlaylist = async () => {
    const trimmed = url.trim();
    if (!trimmed) return;
    setFetching(true);
    setFetchError(null);
    try {
      const data = await apiFetch<PlaylistResponse>(API.download.playlist(trimmed));
      setTitle(data.title);
      setVideos(data.videos.map((v) => ({ ...v, selected: true, status: 'pending', progress: null })));
    } catch (e) {
      setFetchError(`Error: ${errMsg(e)}`);
    } finally {
      setFetching(false);
    }
  };

  const toggleAll = (val: boolean) => setVideos((prev) => prev.map((v) => ({ ...v, selected: val })));
  const toggleOne = (id: string, val: boolean) =>
    setVideos((prev) => prev.map((v) => (v.id === id ? { ...v, selected: val } : v)));

  const startDownload = async () => {
    const selected = videos.filter((v) => v.selected && v.status !== 'completed');
    if (selected.length === 0) {
      toast.warning('No videos selected');
      return;
    }
    setDownloading(true);
    try {
      const res = await apiFetch<{ session_id: string }>(API.download.start, {
        method: 'POST',
        body: {
          videos: selected.map((v) => ({ id: v.id, title: v.title, duration: v.duration, url: v.url })),
          quality,
        },
      });
      setSessionId(res.session_id);
    } catch (e) {
      setDownloading(false);
      toast.error(`Download failed: ${errMsg(e)}`);
    }
  };

  const cancelDownload = async () => {
    if (!sessionId) return;
    try {
      await apiFetch(API.download.cancel(sessionId), { method: 'POST' });
      setDownloading(false);
      toast.warning('Download cancelled');
    } catch (e) {
      toast.error(`Cancel failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-6">
      <PageHeader title="Download" subtitle="Download videos from YouTube playlists" />

      <Card label="Playlist URL">
        <div className="flex gap-3">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && fetchPlaylist()}
            placeholder="https://www.youtube.com/playlist?list=…"
            className="flex-1 rounded-xl border border-border-light bg-surface-100 px-3.5 py-2.5 text-sm text-text-primary transition-all placeholder:text-text-muted focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15"
          />
          <Button intent="primary" onClick={fetchPlaylist} disabled={fetching}>
            {fetching ? 'Fetching…' : 'Fetch Playlist'}
          </Button>
        </div>
        {fetchError && <p className="mt-3 text-sm text-red-400">{fetchError}</p>}
      </Card>

      {videos.length > 0 && (
        <Card>
          <div className="mb-4 flex items-center justify-between gap-4">
            <h3 className="truncate font-heading text-sm font-semibold text-text-primary">{title}</h3>
            <div className="flex items-center gap-3">
              <label className="text-[11px] uppercase tracking-wider text-text-muted">Quality</label>
              <select
                value={quality}
                onChange={(e) => setQuality(e.target.value)}
                disabled={downloading}
                className="cursor-pointer appearance-none rounded-xl border border-border-light bg-surface-100 px-3.5 py-2.5 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15"
              >
                {QUALITIES.map((q) => (
                  <option key={q.value} value={q.value}>
                    {q.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mb-2 flex items-center gap-2 text-xs">
            <Button size="sm" intent="primary" onClick={() => toggleAll(true)}>
              Select All
            </Button>
            <Button size="sm" onClick={() => toggleAll(false)}>
              Deselect All
            </Button>
            <span className="ml-auto font-mono tabular-nums text-text-muted">
              {selectedCount} / {videos.length} selected
            </span>
          </div>

          <div className="max-h-[28rem] space-y-0.5 overflow-y-auto pr-1">
            {videos.map((v) => (
              <div
                key={v.id}
                className={cn(
                  'group flex items-start gap-3 rounded-xl border border-transparent p-3 transition-all duration-200 hover:border-white/5 hover:bg-white/[0.03]',
                  v.status === 'completed' && 'opacity-50',
                )}
              >
                <input
                  type="checkbox"
                  checked={v.selected}
                  disabled={downloading}
                  onChange={(e) => toggleOne(v.id, e.target.checked)}
                  className="mt-0.5 h-3.5 w-3.5 cursor-pointer accent-primary"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm text-text-primary transition-colors group-hover:text-white">{v.title}</p>
                  <div className="mt-1 flex items-center gap-2.5">
                    {v.duration != null && (
                      <span className="font-mono text-[11px] tabular-nums text-text-muted">{formatDuration(v.duration)}</span>
                    )}
                    <StatusBadge status={v.status} />
                  </div>
                  {v.progress && <VideoProgressRow progress={v.progress} completed={v.status === 'completed'} />}
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3 border-t border-border pt-3">
            {downloading ? (
              <Button intent="danger" onClick={cancelDownload}>
                Cancel
              </Button>
            ) : (
              <Button intent="primary" onClick={startDownload}>
                Download Selected
              </Button>
            )}
          </div>
        </Card>
      )}

      <p className="px-1 text-[11px] text-text-muted">
        <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">Enter</kbd> fetch
        playlist
      </p>
    </div>
  );
}

function VideoProgressRow({ progress: p, completed }: { progress: VideoProgress; completed: boolean }) {
  const parts = [
    p.percent != null ? `${p.percent.toFixed(1)}%` : '',
    p.downloaded != null && p.total != null ? `${formatBytes(p.downloaded)}/${formatBytes(p.total)}` : '',
    p.speed ? formatSpeed(p.speed) : '',
    p.eta ? `ETA ${p.eta}s` : '',
  ].filter(Boolean);

  return (
    <div className="mt-2.5">
      <ProgressBar progress={(p.percent ?? 0) / 100} variant={completed ? 'success' : 'primary'} />
      <div className="mt-1.5 flex gap-3 font-mono text-[11px] tabular-nums text-text-muted">
        {parts.map((part, i) => (
          <span key={i}>{part}</span>
        ))}
      </div>
    </div>
  );
}

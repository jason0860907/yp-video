import { useMemo, useState } from 'react';
import { API, ApiError, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { formatBytes, formatDuration, formatSpeed } from '@/lib/format';
import { useSSE } from '@/lib/useSSE';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
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

const fieldCls =
  'w-full rounded-lg border border-border-light bg-surface-50 px-3 py-2.5 text-sm text-text-primary transition-all placeholder:text-text-muted focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';
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
  const completedCount = useMemo(() => videos.filter((v) => v.status === 'completed').length, [videos]);

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

  const hasPlaylist = videos.length > 0;

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="In playlist" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selectedCount} tintClass="text-primary-light" />
        <StatTile label="Completed" value={completedCount} tintClass="text-primary-light" sub={hasPlaylist ? 'this session' : undefined} />
        <StatTile
          label="Status"
          value={downloading ? 'Running' : 'Idle'}
          tintClass={downloading ? 'text-primary-light' : 'text-text-muted'}
        />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        {/* New download form */}
        <Card>
          <SectionLabel>New download · yp-download</SectionLabel>
          <label className="mb-1.5 block text-[10.5px] uppercase tracking-wide text-text-muted">
            YouTube playlist / video URL
          </label>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && fetchPlaylist()}
            placeholder="https://youtube.com/playlist?list=…"
            className={cn(fieldCls, 'mb-3 font-mono')}
          />
          <label className="mb-1.5 block text-[10.5px] uppercase tracking-wide text-text-muted">Quality</label>
          <select value={quality} onChange={(e) => setQuality(e.target.value)} disabled={downloading} className={cn(fieldCls, 'mb-4 cursor-pointer appearance-none')}>
            {QUALITIES.map((q) => (
              <option key={q.value} value={q.value}>
                {q.label}
              </option>
            ))}
          </select>

          <div className="flex flex-wrap items-center gap-2.5">
            <Button intent={hasPlaylist ? 'default' : 'primary'} onClick={fetchPlaylist} disabled={fetching}>
              {fetching ? 'Fetching…' : hasPlaylist ? 'Re-fetch' : 'Fetch Playlist'}
            </Button>
            {hasPlaylist &&
              (downloading ? (
                <Button intent="danger" onClick={cancelDownload}>
                  Cancel
                </Button>
              ) : (
                <Button intent="primary" onClick={startDownload}>
                  Start download
                </Button>
              ))}
          </div>
          {fetchError && <p className="mt-3 text-sm text-red-400">{fetchError}</p>}
        </Card>

        {/* Playlist queue */}
        <Card>
          <div className="mb-2.5 flex items-center justify-between gap-3">
            <SectionLabel className="!mb-0">{hasPlaylist ? title || 'Download queue' : 'Download queue'}</SectionLabel>
            {hasPlaylist && (
              <div className="flex items-center gap-2">
                <Button size="sm" intent="primary" onClick={() => toggleAll(true)}>
                  Select All
                </Button>
                <Button size="sm" onClick={() => toggleAll(false)}>
                  Deselect All
                </Button>
                <span className="font-mono text-[11px] tabular-nums text-text-muted">
                  {selectedCount}/{videos.length}
                </span>
              </div>
            )}
          </div>

          {!hasPlaylist ? (
            <p className="py-10 text-center text-sm text-text-muted">Fetch a playlist to populate the queue.</p>
          ) : (
            <div className="max-h-[30rem] space-y-2 overflow-y-auto pr-1">
              {videos.map((v) => (
                <div
                  key={v.id}
                  className={cn(
                    'rounded-xl border border-border bg-surface-50 px-3.5 py-3 transition-colors',
                    v.status === 'completed' && 'opacity-60',
                  )}
                >
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={v.selected}
                      disabled={downloading}
                      onChange={(e) =>
                        setVideos((prev) => prev.map((x) => (x.id === v.id ? { ...x, selected: e.target.checked } : x)))
                      }
                      className="h-3.5 w-3.5 flex-shrink-0 cursor-pointer accent-primary"
                    />
                    <span className="min-w-0 flex-1 truncate text-[12.5px] text-text-primary">{v.title}</span>
                    {v.duration != null && (
                      <span className="font-mono text-[11px] tabular-nums text-text-muted">{formatDuration(v.duration)}</span>
                    )}
                    <StatusBadge status={v.status} />
                  </div>
                  {v.progress && <VideoProgressRow progress={v.progress} />}
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function VideoProgressRow({ progress: p }: { progress: VideoProgress }) {
  const parts = [
    p.percent != null ? `${p.percent.toFixed(1)}%` : '',
    p.downloaded != null && p.total != null ? `${formatBytes(p.downloaded)}/${formatBytes(p.total)}` : '',
    p.speed ? formatSpeed(p.speed) : '',
    p.eta ? `ETA ${p.eta}s` : '',
  ].filter(Boolean);

  return (
    <div className="mt-2.5 pl-[26px]">
      <ProgressBar progress={(p.percent ?? 0) / 100} />
      <div className="mt-1.5 flex gap-3 font-mono text-[11px] tabular-nums text-text-muted">
        {parts.map((part, i) => (
          <span key={i}>{part}</span>
        ))}
      </div>
    </div>
  );
}

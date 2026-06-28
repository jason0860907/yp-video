import { useEffect, useState } from 'react';
import { API, ApiError, apiPostBlob } from '@/lib/api';
import { cn } from '@/lib/cn';
import { formatTime } from '@/lib/format';
import { downloadBlob } from '@/lib/download';
import { Button } from '@/components/ui/Button';
import { toast } from '@/components/feedback/toast';
import type { EditorAnnotation } from './AnnotationEditor';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

interface Props {
  video: string;
  segments: EditorAnnotation[];
  onClose: () => void;
}

/** Cut the loaded rally annotations into mp4 clips — one at a time, or several
 *  checked at once as a zip. Operates on a snapshot of the segments. */
export function DownloadClipsModal({ video, segments, onClose }: Props) {
  const [picked, setPicked] = useState<Set<number>>(() => new Set(segments.map((s, i) => (s.label === 'rally' ? i : -1)).filter((i) => i >= 0)));
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState('');

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && !busy && onClose();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [busy, onClose]);

  if (!video || !segments.length) return null;

  const toggle = (i: number, on: boolean) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (on) next.add(i);
      else next.delete(i);
      return next;
    });

  const downloadOne = async (i: number) => {
    setBusy(true);
    setStatusText('Cutting clip…');
    try {
      const s = segments[i]!;
      const blob = await apiPostBlob(API.review.clip, { video, segment: s });
      downloadBlob(blob, `${s.label}_${Math.round(s.start)}-${Math.round(s.end)}.mp4`);
    } catch (e) {
      toast.error(`Clip export failed: ${errMsg(e)}`);
    } finally {
      setBusy(false);
      setStatusText('');
    }
  };

  const downloadZip = async () => {
    const chosen = [...picked].sort((a, b) => a - b).map((i) => segments[i]!);
    if (!chosen.length) return;
    setBusy(true);
    setStatusText(`Cutting ${chosen.length} clip(s)…`);
    try {
      const blob = await apiPostBlob(API.review.clipZip, { video, segments: chosen });
      downloadBlob(blob, 'rally-clips.zip');
      toast.success(`Downloaded ${chosen.length} clip(s)`);
    } catch (e) {
      toast.error(`Clip export failed: ${errMsg(e)}`);
    } finally {
      setBusy(false);
      setStatusText('');
    }
  };

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)' }}
      onClick={(e) => e.target === e.currentTarget && !busy && onClose()}
    >
      <div className="flex max-h-[80vh] w-full max-w-lg flex-col rounded-2xl border border-border bg-surface-100 p-6 shadow-2xl">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h3 className="font-heading text-base font-semibold text-text-primary">Download rally clips</h3>
            <p className="mt-1 truncate text-xs text-text-muted">
              Cut from <span className="text-text-secondary">{video.split('/').pop()}</span>
            </p>
          </div>
          <button type="button" onClick={onClose} className="flex-shrink-0 text-text-muted hover:text-text-primary" aria-label="Close">
            <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="mb-2 flex items-center gap-2 text-xs">
          <Button size="sm" onClick={() => setPicked(new Set(segments.map((_, i) => i)))} disabled={busy}>
            Select All
          </Button>
          <Button size="sm" onClick={() => setPicked(new Set())} disabled={busy}>
            Deselect All
          </Button>
          <span className="ml-auto font-mono tabular-nums text-text-muted">
            {picked.size} / {segments.length} selected
          </span>
        </div>
        <div className="flex-1 space-y-0.5 overflow-y-auto border-t border-border pr-1 pt-2">
          {segments.map((s, i) => (
            <div key={i} className="flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-surface-50">
              <input type="checkbox" checked={picked.has(i)} disabled={busy} onChange={(e) => toggle(i, e.target.checked)} className="h-3.5 w-3.5 cursor-pointer accent-primary" />
              <span className="w-5 select-none text-right font-heading text-[10px] text-text-muted/60">{i + 1}</span>
              <span className={cn('rounded-full px-2 py-0.5 text-[10px] font-medium ring-1', s.label === 'rally' ? 'bg-emerald-500/20 text-emerald-400 ring-emerald-500/25' : 'bg-white/[0.06] text-text-muted ring-white/10')}>
                {s.label}
              </span>
              <span className="flex-1 font-mono text-xs tabular-nums text-text-secondary">
                {formatTime(s.start)} → {formatTime(s.end)}
              </span>
              <span className="font-mono text-[10px] tabular-nums text-text-muted">{(s.end - s.start).toFixed(1)}s</span>
              <Button size="sm" onClick={() => downloadOne(i)} disabled={busy}>
                Download
              </Button>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center justify-between gap-2.5 border-t border-border pt-4">
          <span className="text-xs text-text-muted">{statusText}</span>
          <Button intent="primary" onClick={downloadZip} disabled={busy || picked.size === 0}>
            Download ZIP
          </Button>
        </div>
      </div>
    </div>
  );
}

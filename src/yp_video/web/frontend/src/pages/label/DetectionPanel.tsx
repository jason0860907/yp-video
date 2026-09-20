import { useCallback, useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { cn } from '@/lib/cn';
import { Card } from '@/components/ui/Card';
import { fieldCls } from '@/components/form/Field';
import { confirm } from '@/components/feedback/confirm';
import { BoxEditor } from './detection/BoxEditor';
import type { Box } from './detection/BoxEditor';
import type { ModeDescriptor, PlaybackClock, RegisterGuard } from './mode';

export const DETECTION_MODE: ModeDescriptor = {
  key: 'detection',
  label: 'Detection',
  listKey: 'detection-videos',
  statusOptions: [
    { value: 'all', label: 'All' },
    { value: 'unlabeled', label: 'Unlabeled' },
    { value: 'pre-annotate', label: 'Pre-Annotate' },
    { value: 'in-progress', label: 'In-Progress' },
  ],
  status: (row) => row.detection?.status ?? 'unlabeled',
  matches: (row, status) => status === 'all' || (row.detection?.status ?? 'unlabeled') === status,
  available: (row) => Boolean(row.action),
  hint: () => '需要一支 cut 影片',
};
type Annotation = { revision: number; state: 'draft' | 'reviewed'; boxes: Box[] };
type PredictionSource = {
  id: string;
  label: string;
  aligned: boolean;
  first_frame: number | null;
  frame_count: number;
  model: string;
};
type VideoInfo = {
  prediction_sources: PredictionSource[];
  num_frames: number;
  fps: number;
  frames: Record<string, { state: 'draft' | 'reviewed'; count: number }>;
  action_frames: number[];
};
type FrameData = {
  annotation: Annotation | null;
  prediction: {
    boxes: Box[];
    scores: number[];
    frame: number | null;
    message: string;
    source: string | null;
    next_frame: number | null;
    previous_frame: number | null;
  };
};

export function DetectionPanel({
  video,
  registerGuard,
  clock,
}: {
  video: string;
  registerGuard: RegisterGuard;
  clock: PlaybackClock;
}) {
  const info = useQuery({
    queryKey: ['detection-info', video],
    queryFn: () => apiFetch<VideoInfo>(API.detectionLabel.video(video)),
    refetchOnWindowFocus: false,
  });
  const [source, setSource] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0);
  const [frame, setFrame] = useState<number | null>(null);
  useEffect(() => {
    if (!info.data || frame !== null) return;
    const incoming = clock.read(video);
    const first = info.data.prediction_sources.find(s => s.aligned && s.frame_count)?.first_frame ?? 0;
    const initial = Math.max(0, Math.min(info.data.num_frames - 1, incoming !== null ? Math.round(incoming * info.data.fps) : first));
    setFrame(initial);
    clock.write(video, initial / info.data.fps);
  }, [info.data, frame, clock, video]);
  if (info.isError)
    return (
      <Card>
        <p role="alert" className="text-red-400">
          {errMsg(info.error)}
        </p>
        <Button onClick={() => void info.refetch()}>重新載入</Button>
      </Card>
    );
  if (!info.data || frame === null) return <Card>讀取影片…</Card>;
  return (
    <DetectionFrame
      key={`${video}:${frame}:${source}`}
      source={
        source ?? info.data.prediction_sources.find((s) => s.aligned && s.frame_count)?.id ?? ''
      }
      onSource={setSource}
      threshold={threshold}
      onThreshold={setThreshold}
      video={video}
      frame={frame}
      info={info.data}
      registerGuard={registerGuard}
      navigate={(f) => {
        clock.write(video, f / info.data.fps);
        setFrame(f);
      }}
    />
  );
}

function DetectionFrame({
  video,
  frame,
  info,
  navigate,
  registerGuard,
  source,
  onSource,
  threshold,
  onThreshold,
}: {
  source: string;
  onSource: (source: string) => void;
  threshold: number;
  onThreshold: (n: number) => void;
  video: string;
  frame: number;
  info: VideoInfo;
  navigate: (f: number) => void;
  registerGuard: RegisterGuard;
}) {
  const qc = useQueryClient();
  const [data, setData] = useState<FrameData | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [saved, setSaved] = useState<Annotation | null>(null);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [original, setOriginal] = useState(false);
  const [draw, setDraw] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [jump, setJump] = useState(String(frame));
  const [history, setHistory] = useState<Box[][]>([]);
  const saving = useRef(false);
  const load = useCallback(async () => {
    setError('');
    setData(null);
    try {
      const result = await apiFetch<FrameData>(
        API.detectionLabel.frame(video, frame) + (source ? `&source=${source}` : ''),
      );
      setData(result);
      setSaved(result.annotation);
      setBoxes(
        result.annotation?.boxes ??
          result.prediction.boxes.filter(
            (b, i) => (result.prediction.scores[i] ?? 0) >= threshold && b[2] > b[0] && b[3] > b[1],
          ),
      );
      setDirty(false);
      setHistory([]);
      setSelected(null);
    } catch (e) {
      setError(errMsg(e));
    }
    // Threshold is applied to the already loaded predictions when changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video, frame, source]);
  useEffect(() => {
    void load();
  }, [load]);
  const guard = useCallback(async () => {
    if (saving.current) return false;
    return (
      !dirty ||
      (await confirm({
        title: '尚有未儲存的人物框',
        body: '離開會捨棄這次修改。',
        confirmText: '捨棄修改',
        cancelText: '繼續編輯',
        variant: 'warning',
      }))
    );
  }, [dirty]);
  useEffect(() => {
    registerGuard(guard);
    return () => registerGuard(null);
  }, [guard, registerGuard]);
  useEffect(() => {
    const unload = (e: BeforeUnloadEvent) => {
      if (dirty || saving.current) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', unload);
    return () => window.removeEventListener('beforeunload', unload);
  }, [dirty]);
  const change = (next: Box[]) => {
    setHistory((h) => [...h.slice(-49), boxes]);
    setBoxes(next);
    setDirty(true);
  };
  const go = async (f: number) => {
    if (!Number.isInteger(f) || f < 0 || f >= info.num_frames || f === frame) return;
    if (await guard()) navigate(f);
  };
  const save = async (state: Annotation['state']) => {
    if (saving.current || !ready || !data) return;
    if (
      state === 'reviewed' &&
      !boxes.length &&
      !(await confirm({
        title: '確認這個影格完全沒有人？',
        body: '只有確實沒有任何可辨識人物才標為無人；尚未畫框請儲存草稿。',
        confirmText: '確認無人',
        cancelText: '繼續標註',
      }))
    )
      return;
    saving.current = true;
    setBusy(true);
    setError('');
    try {
      const result = await apiFetch<Annotation>(API.detectionLabel.frame(video, frame), {
        method: 'PUT',
        body: { revision: saved?.revision ?? 0, state, boxes },
      });
      setSaved(result);
      setDirty(false);
      void qc.invalidateQueries({ queryKey: ['detection-info', video] });
      void qc.invalidateQueries({ queryKey: ['detection-videos'] });
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
    } catch (e) {
      setError(errMsg(e));
    } finally {
      saving.current = false;
      setBusy(false);
    }
  };
  const reviewed = Object.values(info.frames).filter((f) => f.state === 'reviewed').length;
  const draft = Object.values(info.frames).filter((f) => f.state === 'draft').length;
  const status = dirty
    ? 'Unsaved'
    : saved?.state === 'reviewed'
      ? boxes.length
        ? 'Reviewed'
        : 'Reviewed · empty'
      : saved
        ? 'Draft'
        : boxes.length
          ? 'Pre-annotation'
          : 'Unlabeled';
  const disabled = busy || !data || !ready;
  const candidates =
    data?.prediction.boxes.filter(
      (b, i) => (data.prediction.scores[i] ?? 0) >= threshold && b[2] > b[0] && b[3] > b[1],
    ) ?? [];
  const nextSample = Math.min(info.num_frames - 1, frame + Math.max(1, Math.round(info.fps * 5)));
  const applyPredictions = async () => {
    if (
      (saved || dirty) &&
      !(await confirm({
        title: 'Replace current boxes?',
        body: '目前的框將由所選來源的預測取代，需重新確認。',
        confirmText: 'Replace',
      }))
    )
      return;
    change(candidates);
    setSelected(null);
  };
  const changeThreshold = (value: number) => {
    const n = Math.max(0, Math.min(1, value));
    onThreshold(n);
    if (!saved && !dirty && data)
      setBoxes(
        data.prediction.boxes.filter(
          (b, i) => (data.prediction.scores[i] ?? 0) >= n && b[2] > b[0] && b[3] > b[1],
        ),
      );
  };
  const sourceLabel = info.prediction_sources.find((s) => s.id === source)?.label ?? 'Predictions';
  return (
    <div className="flex flex-col gap-5 lg:flex-row">
      <div className="min-w-0 flex-1 space-y-3">
        <Card>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-heading text-sm font-semibold text-text-primary">Person Labels</h3>
            <div className="flex flex-wrap items-center gap-1.5">
              <Button
                size="sm"
                intent={!draw ? 'primary' : 'default'}
                disabled={disabled}
                onClick={() => setDraw(false)}
              >
                Select
              </Button>
              <Button
                size="sm"
                intent={draw ? 'primary' : 'default'}
                disabled={disabled}
                onClick={() => setDraw(true)}
              >
                Draw box
              </Button>
              <Button
                size="sm"
                disabled={disabled || !history.length}
                onClick={() => {
                  setBoxes(history[history.length - 1]!);
                  setHistory((h) => h.slice(0, -1));
                  setDirty(true);
                  setSelected(null);
                }}
              >
                Undo
              </Button>
              <Button
                size="sm"
                intent="ghost"
                disabled={disabled || selected === null}
                onClick={() => {
                  change(boxes.filter((_, i) => i !== selected));
                  setSelected(null);
                }}
              >
                Delete
              </Button>
            </div>
          </div>
          <div className="overflow-hidden rounded-2xl bg-black ring-1 ring-white/[0.06]">
            <BoxEditor
              key={String(original)}
              src={apiUrl(API.detectionLabel.image(video, frame) + `&original=${original}`)}
              boxes={boxes}
              onChange={change}
              disabled={disabled}
              selected={selected}
              onSelect={setSelected}
              draw={draw}
              onReady={setReady}
            />
          </div>
          <div className="mt-3 space-y-1">
            <input
              aria-label="Frame timeline"
              type="range"
              min={0}
              max={info.num_frames - 1}
              value={jump}
              disabled={busy}
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-surface-300 accent-primary [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
              onChange={(e) => setJump(e.target.value)}
              onPointerUp={(e) => void go(Number(e.currentTarget.value))}
              onKeyUp={(e) => void go(Number(e.currentTarget.value))}
            />
            <div className="flex justify-between font-mono text-[10px] text-text-muted">
              <span>0:00</span>
              <span>
                {Math.floor(info.num_frames / info.fps / 60)}:
                {String(Math.floor(info.num_frames / info.fps) % 60).padStart(2, '0')}
              </span>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="rounded-lg border border-border bg-surface-200/50 px-2.5 py-1 font-mono text-sm tabular-nums text-text-primary">
                {(frame / info.fps).toFixed(2)}s / f{frame}
              </span>
              <Button
                size="sm"
                aria-label="Previous frame"
                disabled={busy || frame === 0}
                onClick={() => void go(frame - 1)}
              >
                ◂
              </Button>
              <Button
                size="sm"
                aria-label="Next frame"
                disabled={busy || frame === info.num_frames - 1}
                onClick={() => void go(frame + 1)}
              >
                ▸
              </Button>
              <Button
                size="sm"
                disabled={busy || nextSample === frame}
                onClick={() => void go(nextSample)}
              >
                +5s
              </Button>
            </div>
            <label className="flex items-center gap-1.5 text-[11px] text-text-muted">
              <input
                type="checkbox"
                checked={original}
                disabled={busy}
                onChange={(e) => {
                  setReady(false);
                  setOriginal(e.target.checked);
                }}
              />
              Original resolution
            </label>
          </div>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-text-muted">
            <span className="font-mono">
              {info.fps.toFixed(3)} fps · {info.num_frames} frames
            </span>
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void go(Number(jump));
              }}
            >
              <label htmlFor="detection-frame">Frame</label>
              <input
                id="detection-frame"
                aria-label="影格編號"
                className={cn(fieldCls, '!w-20 py-1 text-xs')}
                type="number"
                min={0}
                max={info.num_frames - 1}
                value={jump}
                onChange={(e) => setJump(e.target.value)}
              />
              <Button size="sm" disabled={busy}>
                Go
              </Button>
            </form>
          </div>
        </Card>
        <p className="px-1 text-[11px] text-text-muted">
          Draw to add · Drag to move · Drag corners to resize · 所有可辨識人物，僅框可見部分。
        </p>
        {error && (
          <Card>
            <p role="alert" className="text-xs text-red-400">
              {error}
            </p>
            <Button
              size="sm"
              onClick={async () => {
                if (await guard()) void load();
              }}
            >
              Reload annotations
            </Button>
          </Card>
        )}
      </div>
      <div className="min-w-0 lg:w-[360px] lg:flex-shrink-0">
        <Card>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <SectionLabel className="mb-0">
              People ({boxes.length}){dirty ? ' ·' : ''}
            </SectionLabel>
            <div className="flex items-center gap-2">
              <Button size="sm" disabled={disabled} onClick={() => void save('draft')}>
                Save draft
              </Button>
              <Button
                size="sm"
                intent="primary"
                disabled={disabled}
                onClick={() => void save('reviewed')}
              >
                {boxes.length ? 'Confirm' : 'Confirm empty'}
              </Button>
            </div>
          </div>
          <div className="mb-3 flex items-center justify-between gap-2 text-[11px]">
            <span
              role="status"
              className={cn(
                'rounded px-2 py-1',
                saved?.state === 'reviewed' && !dirty
                  ? 'bg-primary/10 text-primary'
                  : 'bg-surface-200 text-text-secondary',
              )}
            >
              {status}
            </span>
            <span className="text-text-muted">
              {reviewed} reviewed · {draft} drafts
            </span>
          </div>
          <div className="border-y border-border py-3 space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-[11px] text-text-muted" htmlFor="prediction-source">
                Source
              </label>
              <select
                id="prediction-source"
                aria-label="Prediction source"
                className={cn(fieldCls, 'min-w-0 flex-1 text-xs')}
                value={source}
                disabled={busy}
                onChange={async (e) => {
                  const value = e.target.value;
                  if (await guard()) onSource(value);
                }}
              >
                {!info.prediction_sources.length && <option value="">No predictions</option>}
                {info.prediction_sources.map((s) => (
                  <option key={s.id} value={s.id} disabled={!s.aligned}>
                    {s.label}
                    {!s.aligned ? ' · misaligned' : ''}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2 text-[11px] text-text-muted">
              <label htmlFor="prediction-score">Min score</label>
              <input
                id="prediction-score"
                aria-label="預測最低分數"
                type="number"
                min={0}
                max={1}
                step={0.05}
                className={cn(fieldCls, '!w-20 py-1 text-xs')}
                value={threshold}
                onChange={(e) => changeThreshold(Number(e.target.value))}
                disabled={busy}
              />
              <Button
                size="sm"
                intent="ghost"
                disabled={disabled || !candidates.length}
                onClick={() => void applyPredictions()}
              >
                Reset to source
              </Button>
            </div>
            <p className="text-[11px] leading-relaxed text-text-muted">
              {saved ? 'Showing saved annotations.' : `${sourceLabel} · unconfirmed`}
              {data?.prediction.frame != null && data.prediction.frame !== frame
                ? ` · sampled at f${data.prediction.frame}`
                : ''}
            </p>
          </div>
          <div className="mt-2 max-h-[340px] overflow-y-auto space-y-1 pr-1">
            {boxes.map((b, i) => (
              <button
                key={i}
                type="button"
                disabled={disabled}
                onClick={() => {
                  setSelected(i);
                  setDraw(false);
                }}
                className={cn(
                  'flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left transition-colors',
                  selected === i
                    ? 'border-primary/40 bg-primary/10 text-text-primary'
                    : 'border-transparent text-text-secondary hover:bg-surface-200',
                )}
              >
                <span className="flex items-center gap-2 text-xs">
                  <span className="h-2 w-2 rounded-full bg-primary" />
                  Person {i + 1}
                </span>
                <span className="font-mono text-[10px] text-text-muted">
                  {Math.round((b[2] - b[0]) * 100)} × {Math.round((b[3] - b[1]) * 100)}%
                </span>
              </button>
            ))}
            {!boxes.length && (
              <div className="py-6 text-center text-xs text-text-muted">
                <p>
                  {saved
                    ? 'No people annotated on this frame.'
                    : (data?.prediction.message ?? 'Loading…')}
                </p>
                <Button
                  size="sm"
                  className="mt-3"
                  disabled={busy || data?.prediction.next_frame == null}
                  onClick={() => void go(data!.prediction.next_frame!)}
                >
                  Next frame with boxes
                </Button>
              </div>
            )}
          </div>
          <div className="mt-3 border-t border-border pt-3 space-y-2">
            <SectionLabel>Frames</SectionLabel>
            <div className="flex gap-2">
              <Button
                size="sm"
                className="flex-1"
                disabled={busy || data?.prediction.previous_frame == null}
                onClick={() => void go(data!.prediction.previous_frame!)}
              >
                Prev boxes
              </Button>
              <Button
                size="sm"
                className="flex-1"
                disabled={busy || data?.prediction.next_frame == null}
                onClick={() => void go(data!.prediction.next_frame!)}
              >
                Next boxes
              </Button>
            </div>
            <select
              aria-label="跳到動作影格"
              value=""
              disabled={busy}
              className={cn(fieldCls, 'text-xs')}
              onChange={(e) => void go(Number(e.target.value))}
            >
              <option value="">Action frames ({info.action_frames.length})</option>
              {info.action_frames.map((f) => (
                <option key={f} value={f}>
                  f{f} · {(f / info.fps).toFixed(2)}s
                </option>
              ))}
            </select>
            <select
              aria-label="已標註影格"
              value=""
              disabled={busy}
              className={cn(fieldCls, 'text-xs')}
              onChange={(e) => void go(Number(e.target.value))}
            >
              <option value="">Labeled frames ({Object.keys(info.frames).length})</option>
              {Object.entries(info.frames)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([f, a]) => (
                  <option key={f} value={f}>
                    f{f} · {a.state} · {a.count} people
                  </option>
                ))}
            </select>
          </div>
          <details className="mt-3 border-t border-border pt-3 text-[11px] text-text-muted">
            <summary className="cursor-pointer">Labeling guide</summary>
            <p className="mt-2 leading-relaxed">
              包含場邊人員，只框可見部分。既有框自動顯示為預標註，修改後先存草稿或確認全部人物；只有已確認影格才是人工真值。空白影格不等於已確認無人。
            </p>
          </details>
        </Card>
      </div>
    </div>
  );
}

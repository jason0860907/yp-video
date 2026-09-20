import { useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/Button';
import { fieldCls } from '@/components/form/Field';
import type { Clip, Preview } from './types';
import { KIND_LABEL } from './types';

export function ResultPreview({
  data,
  media,
  local = false,
}: {
  data: Preview;
  media: string;
  local?: boolean;
}) {
  const [scope, setScope] = useState<'rallies' | 'actions' | 'scores'>('rallies');
  const [kind, setKind] = useState('');
  const [player, setPlayer] = useState('');
  const [inRally, setInRally] = useState(true);
  const [rally, setRally] = useState('');
  const [selected, setSelected] = useState('');
  const [autoNext, setAutoNext] = useState(false);
  const [error, setError] = useState('');
  const pendingPlay = useRef(false);
  const video = useRef<HTMLVideoElement>(null);
  const clips = useMemo(
    () =>
      data[scope].filter((c) => {
        const playerMatches = (number: number | undefined) => !player || String(number) === player;
        const touchMatches = c.touches.some(
          (t) => (!kind || t.kind === kind) && playerMatches(t.player?.number),
        );
        if (scope === 'rallies') return (!kind && !player) || touchMatches;
        if (inRally && c.rally_index == null) return false;
        if (!playerMatches(c.player?.number)) return false;
        if (scope === 'scores') return !kind || c.touches.some((t) => t.kind === kind);
        return (!kind || c.kind === kind) && (!rally || String(c.rally_index) === rally);
      }),
    [data, scope, player, kind, inRally, rally],
  );
  const active = clips.find((c) => c.key === selected) ?? clips[0];
  const losses = clips.filter((c) => c.result === 'loss');
  const reasons = losses.reduce<Record<string, number>>((counts, c) => {
    const reason = c.loss_reason || '未分類';
    counts[reason] = (counts[reason] || 0) + 1;
    return counts;
  }, {});
  const playerLabel = (p: { number: number; name: string }) =>
    local ? p.name : `#${p.number} ${p.name}`;
  const play = (c: Clip) => {
    pendingPlay.current = active?.key !== c.key;
    setSelected(c.key);
    if (video.current) {
      video.current.currentTime = c.start;
      void video.current.play().catch(() => setError('無法播放影片，請確認來源可讀取。'));
    }
  };
  return (
    <div className="space-y-4">
      {data.warnings.map((w) => (
        <p role="status" key={w} className="text-sm text-amber-400">
          {w}
        </p>
      ))}
      <div className="flex flex-wrap gap-2">
        {(['rallies', 'actions', 'scores'] as const).map((s) => (
          <Button
            key={s}
            intent={s === scope ? 'primary' : 'default'}
            onClick={() => {
              setScope(s);
              setSelected('');
              video.current?.pause();
            }}
          >
            {s === 'rallies' ? 'Rally' : s === 'actions' ? 'Action' : 'Score'}
          </Button>
        ))}
      </div>
      <div className="grid gap-2 sm:grid-cols-4">
        <select
          aria-label="球員篩選"
          className={fieldCls}
          value={player}
          onChange={(e) => setPlayer(e.target.value)}
        >
          <option value="">全部球員</option>
          {data.roster.map((p) => (
            <option key={p.number} value={p.number}>
              {playerLabel(p)}
            </option>
          ))}
        </select>
        <select
          aria-label="動作篩選"
          className={fieldCls}
          value={kind}
          onChange={(e) => setKind(e.target.value)}
        >
          <option value="">全部動作</option>
          {['serve', 'receive', 'set', 'spike', 'block'].map((k) => (
            <option key={k} value={k}>
              {KIND_LABEL[k]}
            </option>
          ))}
        </select>
        {scope !== 'rallies' && (
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={inRally}
              onChange={(e) => setInRally(e.target.checked)}
            />
            僅 In Rally
          </label>
        )}
        {scope === 'actions' && (
          <select
            aria-label="回合篩選"
            className={fieldCls}
            value={rally}
            onChange={(e) => setRally(e.target.value)}
          >
            <option value="">全部回合</option>
            {data.rallies.map((r) => (
              <option key={r.key} value={r.rally_index ?? ''}>
                Rally {r.rally_index}
              </option>
            ))}
          </select>
        )}
      </div>
      {scope === 'scores' && (
        <div className="rounded-lg bg-surface-100 p-3 text-sm space-y-2">
          <p>
            篩選後 {clips.length} 個片段，失分 {losses.length} 次。
            {Object.entries(reasons)
              .map(([r, n]) => `${r}：${n}`)
              .join(' ／ ')}
          </p>
          {Object.entries(data.court_totals).map(([set, counts]) => (
            <p key={set}>
              第 {set} 局（完整回合累計）：
              {Object.entries(counts)
                .map(([side, n]) => `${side} ${n}`)
                .join(' ／ ')}
            </p>
          ))}
          <p className="text-text-muted">場邊累計不代表換場後的球隊比分。</p>
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-2">
          {active && media ? (
            <video
              key={`${media}:${scope}:${active.key}:${active.start}:${active.end}`}
              ref={video}
              className="w-full rounded-lg bg-black"
              controls
              preload="metadata"
              src={media}
              onLoadedMetadata={(e) => {
                setError('');
                e.currentTarget.currentTime = active.start;
                if (pendingPlay.current) {
                  pendingPlay.current = false;
                  void e.currentTarget.play().catch(() => setError('無法自動播放，請按播放鍵。'));
                }
              }}
              onError={() => setError('影片讀取失敗，請確認雲端來源或選擇對應的 pipeline 影片。')}
              onTimeUpdate={(e) => {
                if (e.currentTarget.currentTime >= active.end) {
                  e.currentTarget.pause();
                  const i = clips.indexOf(active);
                  const next = clips[i + 1];
                  if (autoNext && next) play(next);
                }
              }}
            />
          ) : (
            <p className="text-sm text-text-muted">沒有可播放的片段或影片來源。</p>
          )}
          {error && (
            <p role="alert" className="text-red-400">
              {error}
            </p>
          )}
          {active && (
            <>
              <p className="text-sm">
                {active.start.toFixed(3)}–{active.end.toFixed(3)} 秒 · {active.key}
              </p>
              <div className="flex flex-wrap gap-2">
                {active.touches.map((t, i) => (
                  <Button
                    key={`${t.event_id}:${i}`}
                    size="sm"
                    onClick={() => {
                      if (video.current) video.current.currentTime = t.time;
                    }}
                  >
                    {KIND_LABEL[t.kind]} {t.time.toFixed(2)} {t.player ? playerLabel(t.player) : ''}
                  </Button>
                ))}
              </div>
              <label className="flex gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={autoNext}
                  onChange={(e) => setAutoNext(e.target.checked)}
                />
                自動播放下一段
              </label>
            </>
          )}
        </div>
        <div className="max-h-[560px] space-y-2 overflow-auto">
          <p className="text-sm text-text-muted">{clips.length} 個片段</p>
          {clips.map((c, i) => (
            <button
              key={`${c.key}:${i}`}
              className={`w-full rounded-lg border p-3 text-left text-sm ${active === c ? 'border-primary bg-primary/10' : 'border-border-light'}`}
              onClick={() => play(c)}
            >
              <span className="font-semibold">
                {KIND_LABEL[c.kind]} {i + 1}
              </span>{' '}
              · {c.start.toFixed(2)}–{c.end.toFixed(2)} 秒
              <span className="block text-text-secondary">
                {c.rally_index ? `Rally ${c.rally_index}` : '回合外'}{' '}
                {c.player ? playerLabel(c.player) : ''}{' '}
                {c.result === 'loss' ? '失分' : c.result === 'point' ? '得分' : ''}{' '}
                {c.loss_reason || ''}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

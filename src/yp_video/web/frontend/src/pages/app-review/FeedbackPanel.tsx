/** Review the user's corrections one by one. A review is pinned to the exact
 *  App snapshot it judges; only an explicit, evidenced verdict edits a human
 *  annotation, everything else stays feedback. */

import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { API, apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Collapsible } from '@/components/ui/Collapsible';
import type { Candidate, LibraryMatch, MatchDetail, Review } from './types';
import { DECISION_LABEL, SIDE_LABEL, fmtDate, fmtTime, lossReasonLabel } from './types';

type Value = Record<string, unknown>;

function title(c: Candidate): string {
  const v = c.value as Value;
  const [scope, rest] = [c.scope, c.id.slice(c.id.indexOf(':') + 1)];
  if (scope === 'actions' || scope === 'scores') {
    const parts = [
      v.removed && '刪除',
      v.trim_start != null &&
        `裁切 ${fmtTime(v.trim_start as number)}–${fmtTime(v.trim_end as number)}`,
      v.loss_reason && `失分：${lossReasonLabel(v.loss_reason as string)}`,
      (v.tag_ids as string[]).length > 0 && '收藏',
    ].filter(Boolean);
    const where =
      rest.startsWith('rally-') || rest.startsWith('unmapped-')
        ? '回合結尾'
        : fmtTime(Number(rest));
    return `${scope === 'actions' ? 'Action' : 'Score'} ${where}：${parts.join('、') || '無變更'}`;
  }
  if (scope === 'rally') return `刪除 Rally ${rest}`;
  if (scope === 'rally_bounds') {
    const [b, a] = [v.before as [number, number], v.after as [number, number]];
    return `Rally ${rest} 界線 ${fmtTime(b[0])}–${fmtTime(b[1])} → ${fmtTime(a[0])}–${fmtTime(a[1])}`;
  }
  if (scope === 'winner') {
    const side = (s: unknown) => (s ? (SIDE_LABEL[s as string] ?? String(s)) : '未知');
    return `Rally ${rest} 得分方 ${side(v.before)} → ${side(v.after)}`;
  }
  if (scope === 'player') return `觸球 ${v.event_id as string} 指派 #${v.number as number}`;
  if (scope === 'identification') return '人物分組與門檻';
  if (scope === 'roster') return '球員名單';
  return c.id;
}

/** Operator-published matches stream a pipeline cut; that cut is the label target. */
function sourceCut(match: LibraryMatch): string {
  const key = match.source_video?.public_url ? null : match.source_video?.r2_key;
  return key?.split('/').at(-1) ?? '';
}

export function FeedbackPanel({
  user,
  detail,
  onSeek,
}: {
  user: string;
  detail: MatchDetail;
  onSeek: (clip: { start: number; end: number }) => void;
}) {
  const queryClient = useQueryClient();
  const [reviewId, setReviewId] = useState(detail.review_id);
  const [selected, setSelected] = useState('');
  const [note, setNote] = useState('');
  const [video, setVideo] = useState(() => sourceCut(detail.match));
  const [sameSource, setSameSource] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const review = useQuery({
    queryKey: ['app-review', 'review', reviewId],
    enabled: !!reviewId,
    queryFn: () => apiFetch<Review>(API.appReview.review(reviewId!)),
  });
  const videos = useQuery({
    queryKey: ['app-review', 'videos'],
    queryFn: () => apiFetch<{ name: string }[]>(API.appReview.videos),
  });
  const target = useQuery({
    queryKey: ['app-review', 'target', video],
    enabled: !!video,
    queryFn: () => apiFetch<Record<string, string | null>>(API.appReview.target(video)),
  });

  const candidates = review.data?.candidates ?? detail.candidates;
  const candidate = candidates.find((c) => c.id === selected);
  const decisions = review.data?.decisions ?? {};
  const previous = candidate ? decisions[candidate.id] : undefined;
  const operation = candidate?.can_remove_event
    ? 'remove_event'
    : candidate?.scope === 'rally_bounds'
      ? 'update_rally'
      : 'remove_rally';
  const canApply =
    !!candidate &&
    (candidate.can_remove_event ||
      ((candidate.scope === 'rally' || candidate.scope === 'rally_bounds') && !!candidate.rally));
  const locked =
    busy ||
    !review.data ||
    review.data.application != null ||
    ['remove_event', 'remove_rally', 'update_rally'].includes(previous?.decision ?? '');

  const run = async (fn: () => Promise<Review>) => {
    setBusy(true);
    setError('');
    try {
      const next = await fn();
      queryClient.setQueryData(['app-review', 'review', next.id], next);
      setReviewId(next.id);
      void queryClient.invalidateQueries({ queryKey: ['app-review', 'match'] });
      void target.refetch();
      setNote('');
      setSameSource(false);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  const decide = (decision: string) =>
    run(() =>
      apiFetch<Review>(API.appReview.decisions(reviewId!), {
        method: 'POST',
        body: {
          candidate_id: candidate!.id,
          decision,
          note,
          video,
          revision: target.data?.[decision] || '',
          same_source: sameSource,
        },
      }),
    );

  if (candidates.length === 0)
    return <p className="text-sm text-text-muted">此比賽沒有使用者修正。</p>;

  return (
    <div className="flex min-h-0 flex-col gap-3 text-sm">
      {!reviewId ? (
        <div className="space-y-2 rounded-lg border border-border-light p-3">
          <p className="text-text-secondary">
            {candidates.length} 項修正尚未審核。開始審核會固定目前這份 App
            快照，之後使用者再改也不影響。
          </p>
          <Button
            intent="primary"
            disabled={busy}
            onClick={() =>
              void run(() =>
                apiFetch<Review>(API.appReview.startReview(user, detail.match.id), {
                  method: 'POST',
                }),
              )
            }
          >
            開始審核
          </Button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
          已審核 {Object.keys(decisions).length}/{candidates.length}
          {detail.reviews.length > 1 && (
            <select
              aria-label="審核快照"
              className={cn(fieldCls, 'h-8 w-auto py-0 text-xs')}
              value={reviewId}
              onChange={(e) => setReviewId(e.target.value)}
            >
              {detail.reviews.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id === detail.review_id ? '目前快照' : '舊快照'} ·{' '}
                  {fmtDate(Date.parse(r.created_at) / 1000)} · {r.reviewed}/{r.total}
                </option>
              ))}
            </select>
          )}
        </div>
      )}
      {review.data?.application != null && (
        <Button
          disabled={busy}
          onClick={() =>
            void run(() => apiFetch<Review>(API.appReview.recover(reviewId!), { method: 'POST' }))
          }
        >
          恢復中斷的標註匯入
        </Button>
      )}

      <div className="max-h-72 space-y-1 overflow-auto pr-1">
        {candidates.map((c) => {
          const d = decisions[c.id];
          return (
            <button
              key={c.id}
              onClick={() => {
                setSelected(c.id);
                setNote('');
                setSameSource(false);
                setError('');
                if (c.clip) onSeek(c.clip);
              }}
              className={cn(
                'w-full rounded-lg border px-3 py-2 text-left',
                c.id === selected
                  ? 'border-primary/60 bg-primary/10'
                  : 'border-border-light hover:border-border-bright',
              )}
            >
              <span className="block">{title(c)}</span>
              <span className="mt-1 block">
                {d ? (
                  <Badge tone="success">{DECISION_LABEL[d.decision] ?? d.decision}</Badge>
                ) : (
                  <Badge>待審核</Badge>
                )}
              </span>
            </button>
          );
        })}
      </div>

      {candidate && (
        <div className="space-y-3 border-t border-border-light pt-3">
          {candidate.reason && <p className="text-xs text-amber-400">{candidate.reason}</p>}
          <Collapsible label="原始修正內容">
            <pre className="max-h-48 overflow-auto rounded bg-surface-100 p-2 text-xs">
              {JSON.stringify(candidate.value, null, 2)}
            </pre>
          </Collapsible>
          {previous && (
            <p className="text-xs text-text-muted">
              {DECISION_LABEL[previous.decision] ?? previous.decision} · {previous.actor} ·{' '}
              {previous.at}
              {previous.note && <span className="block text-text-secondary">{previous.note}</span>}
            </p>
          )}
          <textarea
            aria-label="審核依據"
            className={fieldCls}
            placeholder="記錄影片中的觀察、誤判依據或不採用的原因"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={locked} onClick={() => void decide('accepted_feedback')}>
              保留為已審核回饋
            </Button>
            <Button size="sm" disabled={locked} onClick={() => void decide('rejected')}>
              不採用
            </Button>
          </div>
          {canApply && (
            <div className="space-y-2 rounded-lg border border-border-light p-3">
              <label className="block text-xs text-text-muted">
                寫入的 pipeline 影片
                <select
                  className={fieldCls}
                  value={video}
                  onChange={(e) => setVideo(e.target.value)}
                >
                  <option value="">選擇影片</option>
                  {videos.data?.map((v) => (
                    <option key={v.name} value={v.name}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={sameSource}
                  onChange={(e) => setSameSource(e.target.checked)}
                />
                已對照兩份影片，確認是同一來源、同一時間軸，且此修改符合標註定義，而非使用者觀看偏好。
              </label>
              <Button
                size="sm"
                intent="danger"
                disabled={locked || !sameSource || !note.trim() || !target.data?.[operation]}
                onClick={() => void decide(operation)}
              >
                {operation === 'update_rally'
                  ? '確認完整回合界線並更新標註'
                  : '確認誤判並移除該人工標註'}
              </Button>
              {video && target.data && !target.data[operation] && (
                <p className="text-xs text-amber-400">
                  尚無人工標註，請先到 Label 完整審核並儲存。
                </p>
              )}
              {video && (
                <div className="flex flex-wrap gap-3 text-xs text-primary">
                  {(
                    [
                      ['rally', 'Rally'],
                      ['action', 'Action'],
                      ['association', 'Association'],
                      ['reid', 'ReID'],
                    ] as const
                  ).map(([mode, label]) => (
                    <Link
                      key={mode}
                      target="_blank"
                      to={`/label?video=${encodeURIComponent(video)}&mode=${mode}`}
                    >
                      開啟 {label} 標註
                    </Link>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
      {(error || review.error || target.error) && (
        <p role="alert" className="text-xs text-red-400">
          {error || errMsg(review.error ?? target.error)}
        </p>
      )}
    </div>
  );
}

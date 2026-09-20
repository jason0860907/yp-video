import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { API, apiFetch, errMsg } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { fieldCls } from '@/components/form/Field';
import type { Review } from './types';

export function FeedbackPanel({
  review,
  video,
  media,
  onChange,
}: {
  review: Review;
  video: string;
  media: string;
  onChange: (r: Review) => void;
}) {
  const [selected, setSelected] = useState('');
  const [note, setNote] = useState('');
  const [sameSource, setSameSource] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const candidate = review.candidates.find((c) => c.id === selected) ?? review.candidates[0];
  const operation = candidate?.can_remove_event
    ? 'remove_event'
    : candidate?.scope === 'rally_bounds'
      ? 'update_rally'
      : 'remove_rally';
  const canApply =
    !!candidate &&
    (candidate.can_remove_event ||
      ((candidate.scope === 'rally' || candidate.scope === 'rally_bounds') && !!candidate.rally));
  const previous = candidate ? review.decisions[candidate.id] : undefined;
  const alreadyApplied = ['remove_event', 'remove_rally', 'update_rally'].includes(
    previous?.decision ?? '',
  );
  const target = useQuery({
    queryKey: ['app-review-target', video],
    enabled: !!video,
    queryFn: () => apiFetch<Record<string, string | null>>(API.appReview.target(video)),
  });
  const apply = async (decision: string) => {
    if (!candidate) return;
    setBusy(true);
    setError('');
    try {
      const next = await apiFetch<Review>(API.appReview.decisions(review.id), {
        method: 'POST',
        body: {
          candidate_id: candidate.id,
          decision,
          note,
          video,
          revision: target.data?.[decision] || '',
          same_source: sameSource,
        },
      });
      onChange(next);
      setNote('');
      setSameSource(false);
      void target.refetch();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  const recover = async () => {
    setBusy(true);
    setError('');
    try {
      onChange(await apiFetch<Review>(API.appReview.recover(review.id), { method: 'POST' }));
      void target.refetch();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="space-y-3 border-t border-border-light pt-5">
      <h2 className="text-lg font-semibold">修正審核</h2>
      <p className="text-sm text-text-secondary">
        先觀看原始片段，再保存審核意見。「確認誤判」才會移除既有人工標註中的事件或回合；裁切、收藏、得失分與人物分組不會自動成為訓練答案。
      </p>
      {review.application != null && (
        <Button disabled={busy} onClick={() => void recover()}>
          恢復中斷的標註匯入
        </Button>
      )}
      {!candidate ? (
        <p>此快照沒有待審核項目。</p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="max-h-[650px] space-y-2 overflow-auto">
            {review.candidates.map((c) => (
              <button
                key={c.id}
                className={`w-full rounded border p-3 text-left text-sm ${c === candidate ? 'border-primary' : 'border-border-light'}`}
                onClick={() => {
                  setSelected(c.id);
                  setNote('');
                  setSameSource(false);
                  setError('');
                }}
              >
                {c.id}
                <span className="block text-text-muted">
                  {review.decisions[c.id]?.decision || '待審核'}
                </span>
              </button>
            ))}
          </div>
          <div className="space-y-3">
            {candidate.clip && media && (
              <video
                key={`${review.id}:${candidate.id}:${media}`}
                className="w-full rounded bg-black"
                controls
                preload="metadata"
                src={media}
                onLoadedMetadata={(e) => {
                  e.currentTarget.currentTime = candidate.clip!.start;
                }}
                onTimeUpdate={(e) => {
                  if (e.currentTarget.currentTime >= candidate.clip!.end) e.currentTarget.pause();
                }}
              />
            )}
            <pre className="max-h-48 overflow-auto rounded bg-surface-100 p-3 text-xs">
              {JSON.stringify(candidate.value, null, 2)}
            </pre>
            {candidate.reason && <p className="text-sm text-amber-400">{candidate.reason}</p>}
            {previous && (
              <p className="text-sm text-text-muted">
                審核者：{previous.actor} · {previous.at}
                <br />
                {previous.note}
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
              <Button
                disabled={busy || alreadyApplied || review.application != null}
                onClick={() => void apply('accepted_feedback')}
              >
                保留為已審核回饋
              </Button>
              <Button
                disabled={busy || alreadyApplied || review.application != null}
                onClick={() => void apply('rejected')}
              >
                不採用
              </Button>
            </div>
            {canApply && (
              <div className="space-y-3 rounded border border-border-light p-3">
                <p className="text-sm">標註目標：{video || '請先選擇對應的 pipeline 影片'}</p>
                <label className="flex gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={sameSource}
                    onChange={(e) => setSameSource(e.target.checked)}
                  />
                  已對照兩份影片，確認是同一來源、同一時間軸，且此修改符合標註定義，而非使用者觀看偏好。
                </label>
                <Button
                  intent="danger"
                  disabled={
                    busy ||
                    alreadyApplied ||
                    !sameSource ||
                    !note.trim() ||
                    !target.data?.[operation] ||
                    review.application != null
                  }
                  onClick={() => void apply(operation)}
                >
                  {operation === 'update_rally'
                    ? '確認完整回合界線並更新標註'
                    : '確認誤判並移除該人工標註'}
                </Button>
                <Button
                  size="sm"
                  disabled={!video || busy}
                  onClick={() => {
                    setSameSource(false);
                    void target.refetch();
                  }}
                >
                  重新載入目標版本
                </Button>
                {video && !target.data?.[operation] && (
                  <p className="text-sm text-amber-400">
                    尚無人工標註，請先到 Label 完整審核並儲存。
                  </p>
                )}
              </div>
            )}
            {video && (
              <div className="flex flex-wrap gap-3 text-sm text-primary">
                {['rally', 'action', 'association', 'reid'].map((mode) => (
                  <Link
                    key={mode}
                    target="_blank"
                    to={`/label?video=${encodeURIComponent(video)}&mode=${mode}`}
                  >
                    開啟 {mode} 標註
                  </Link>
                ))}
              </div>
            )}
            {(error || target.error) && (
              <p role="alert" className="text-red-400">
                {error || errMsg(target.error)}
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

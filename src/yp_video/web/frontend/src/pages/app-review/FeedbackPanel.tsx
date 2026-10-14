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
import { ErrorState } from '@/components/ui/ErrorState';
import { toast } from '@/components/feedback/toast';
import { EventRows } from '@/components/action/RallyRows';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { formatActionTime } from '@/lib/actionEditorModel';
import { formatDateTime } from '@/lib/format';
import type { Candidate, LibraryMatch, MatchDetail, Review } from './types';
import { DECISION_LABEL, lossReasonLabel, sideLabel } from './types';

type Value = Record<string, unknown>;

const LABEL_MODES = [
  ['rally', 'Rally'],
  ['action', 'Action'],
  ['association', 'Association'],
  ['reid', 'ReID'],
] as const;

function title(c: Candidate): string {
  const v = c.value as Value;
  const [scope, rest] = [c.scope, c.id.slice(c.id.indexOf(':') + 1)];
  const span = (a: [number, number]) => `${formatActionTime(a[0])}–${formatActionTime(a[1])}`;
  if (scope === 'actions' || scope === 'scores') {
    const parts = [
      v.removed && 'removed',
      v.trim_start != null && `trimmed ${span([v.trim_start as number, v.trim_end as number])}`,
      v.loss_reason && `loss: ${lossReasonLabel(v.loss_reason as string)}`,
      (v.tag_ids as string[]).length > 0 && 'favorited',
    ].filter(Boolean);
    const where =
      rest.startsWith('rally-') || rest.startsWith('unmapped-') ? 'rally end' : formatActionTime(Number(rest));
    return `${scope === 'actions' ? 'Action' : 'Score'} ${where}: ${parts.join(', ') || 'no change'}`;
  }
  if (scope === 'rally') return `Rally ${rest} deleted`;
  if (scope === 'rally_bounds')
    return `Rally ${rest} bounds ${span(v.before as [number, number])} → ${span(v.after as [number, number])}`;
  if (scope === 'winner') {
    const side = (s: unknown) => (s ? sideLabel(String(s)) : 'unknown');
    return `Rally ${rest} winner ${side(v.before)} → ${side(v.after)}`;
  }
  if (scope === 'player') return `Touch ${v.event_id as string} assigned #${v.number as number}`;
  if (scope === 'identification') return 'Person grouping and thresholds';
  if (scope === 'roster') return 'Roster';
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

  const run = async (fn: () => Promise<Review>, done: string) => {
    setBusy(true);
    try {
      const next = await fn();
      queryClient.setQueryData(['app-review', 'review', next.id], next);
      setReviewId(next.id);
      void queryClient.invalidateQueries({ queryKey: ['app-review', 'match'] });
      void target.refetch();
      setNote('');
      setSameSource(false);
      toast.success(done);
    } catch (e) {
      toast.error(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  const decide = (decision: string) =>
    run(
      () =>
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
      DECISION_LABEL[decision] ?? decision,
    );

  if (candidates.length === 0)
    return <p className="py-6 text-center text-xs text-text-muted">This match has no user corrections.</p>;
  if (review.error) return <ErrorState message={errMsg(review.error)} onRetry={() => void review.refetch()} />;

  return (
    <div className="flex min-h-0 flex-col gap-3 text-sm">
      {!reviewId ? (
        <div className="space-y-2 rounded-xl border border-border bg-surface-100 p-3">
          <p className="text-xs text-text-secondary">
            {candidates.length} corrections not reviewed. Starting a review pins this App snapshot — later
            edits by the user do not change it.
          </p>
          <Button
            size="sm"
            intent="primary"
            disabled={busy}
            onClick={() =>
              void run(
                () => apiFetch<Review>(API.appReview.startReview(user, detail.match.id), { method: 'POST' }),
                'Review started',
              )
            }
          >
            Start review
          </Button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
          Reviewed {Object.keys(decisions).length}/{candidates.length}
          {detail.reviews.length > 1 && (
            <select
              aria-label="Review snapshot"
              className={cn(fieldCls, 'h-8 w-auto py-0 text-xs')}
              value={reviewId}
              onChange={(e) => setReviewId(e.target.value)}
            >
              {detail.reviews.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id === detail.review_id ? 'Current snapshot' : 'Older snapshot'} ·{' '}
                  {formatDateTime(Date.parse(r.created_at) / 1000)} · {r.reviewed}/{r.total}
                </option>
              ))}
            </select>
          )}
        </div>
      )}
      {review.data?.application != null && (
        <Button
          size="sm"
          disabled={busy}
          onClick={() =>
            void run(
              () => apiFetch<Review>(API.appReview.recover(reviewId!), { method: 'POST' }),
              'Interrupted label import recovered',
            )
          }
        >
          Recover the interrupted label import
        </Button>
      )}

      <div className="vq-list max-h-72 overflow-y-auto pr-1 [&>div]:ml-0">
        <EventRows
          entries={candidates}
          empty="No corrections"
          columns="minmax(0,1fr) auto"
          selectedId={selected || null}
          isActive={() => false}
          onJump={(c) => {
            setSelected(c.id);
            setNote('');
            setSameSource(false);
            if (c.clip) onSeek(c.clip);
          }}
        >
          {(c) => {
            const d = decisions[c.id];
            return (
              <>
                <span className="text-xs text-text-primary">{title(c)}</span>
                {d ? (
                  <Badge tone="success">{DECISION_LABEL[d.decision] ?? d.decision}</Badge>
                ) : (
                  <Badge>To review</Badge>
                )}
              </>
            );
          }}
        </EventRows>
      </div>

      {candidate && (
        <div className="space-y-3 border-t border-border pt-3">
          {candidate.reason && <p className="text-xs text-amber-400">{candidate.reason}</p>}
          <Collapsible label="Raw correction">
            <pre className="max-h-48 overflow-auto rounded bg-surface-100 p-2 text-xs">
              {JSON.stringify(candidate.value, null, 2)}
            </pre>
          </Collapsible>
          {previous && (
            <p className="text-xs text-text-muted">
              {DECISION_LABEL[previous.decision] ?? previous.decision} · {previous.actor} · {previous.at}
              {previous.note && <span className="block text-text-secondary">{previous.note}</span>}
            </p>
          )}
          <textarea
            aria-label="Review note"
            className={fieldCls}
            placeholder="What the video shows, why it is a misdetection, or why the change is not adopted"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={locked} onClick={() => void decide('accepted_feedback')}>
              Keep as reviewed feedback
            </Button>
            <Button size="sm" disabled={locked} onClick={() => void decide('rejected')}>
              Reject
            </Button>
          </div>
          {canApply && (
            <div className="space-y-2 rounded-xl border border-border bg-surface-100 p-3">
              <p className="text-xs text-text-muted">Pipeline video to write</p>
              <VideoCombobox
                items={videos.data ?? []}
                value={video}
                onChange={setVideo}
                query={videos}
                placeholder="Search pipeline videos…"
              />
              <label className="flex gap-2 text-xs">
                <input type="checkbox" checked={sameSource} onChange={(e) => setSameSource(e.target.checked)} />
                I compared both videos: same source, same timeline, and the change fits the label definition
                rather than a viewing preference.
              </label>
              <Button
                size="sm"
                intent="danger"
                disabled={locked || !sameSource || !note.trim() || !target.data?.[operation]}
                onClick={() => void decide(operation)}
              >
                {operation === 'update_rally'
                  ? 'Confirm the full rally bounds and update the label'
                  : 'Confirm the misdetection and remove the human label'}
              </Button>
              {video && target.data && !target.data[operation] && (
                <p className="text-xs text-amber-400">No human label yet — fully review and save it in Label first.</p>
              )}
              {target.error && <p className="text-xs text-red-400">{errMsg(target.error)}</p>}
              {video && (
                <div className="flex flex-wrap gap-3 text-xs text-primary">
                  {LABEL_MODES.map(([mode, label]) => (
                    <Link key={mode} target="_blank" to={`/label?video=${encodeURIComponent(video)}&mode=${mode}`}>
                      Open {label} label
                    </Link>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

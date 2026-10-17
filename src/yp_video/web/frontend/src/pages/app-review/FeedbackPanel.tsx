/** Review the user's corrections one by one, grouped by the rally they
 *  touch. The first verdict pins the exact App snapshot it judges; only an
 *  explicit, evidenced write-back edits a human annotation, everything else
 *  stays feedback. */

import { useEffect, useState } from 'react';
import { useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query';
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
import type { Candidate, Decision, LibraryMatch, MatchDetail, Review } from './types';
import { byReviewOrder, DECISION_LABEL, lossReasonLabel, preciseTime, sideLabel } from './types';

type Value = Record<string, unknown>;
type Operation = 'remove_event' | 'remove_rally' | 'update_rally';

const LABEL_MODES = [
  ['rally', 'Rally'],
  ['action', 'Action'],
  ['association', 'Association'],
  ['reid', 'ReID'],
] as const;

const WRITE_BACK: Record<Operation, string> = {
  remove_event: 'Remove the action from the label',
  remove_rally: 'Remove the rally from the label',
  update_rally: 'Set the label rally to these bounds',
};

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** What the user changed, without the rally — the group header says that. */
function title(c: Candidate): string {
  const v = c.value as Value;
  const span = (a: [number, number]) => `${formatActionTime(a[0])}–${formatActionTime(a[1])}`;
  if (c.scope === 'actions' || c.scope === 'scores') {
    const parts = [
      v.removed && 'removed',
      v.trim_start != null && `trimmed to ${span([v.trim_start as number, v.trim_end as number])}`,
      v.loss_reason && `loss reason ${lossReasonLabel(v.loss_reason as string)}`,
      (v.tag_ids as string[]).length > 0 && 'favorited',
    ].filter(Boolean);
    const at = c.clip?.anchor_time;
    const what = `${c.scope === 'scores' ? 'Point' : capitalize(c.clip?.kind ?? 'action')}${at != null ? ` ${preciseTime(at)}` : ''}`;
    return `${what}: ${parts.join(', ') || 'no change'}`;
  }
  if (c.scope === 'rally') return 'Rally deleted';
  if (c.scope === 'rally_bounds')
    return `Bounds ${span(v.before as [number, number])} → ${span(v.after as [number, number])}`;
  if (c.scope === 'winner') {
    const side = (s: unknown) => (s ? sideLabel(String(s)) : 'unknown');
    return `Winner ${side(v.before)} → ${side(v.after)}`;
  }
  // A player candidate's span is the touch ± 1 s.
  if (c.scope === 'player')
    return `Touch${c.clip ? ` ${preciseTime(c.clip.end - 1)}` : ''} assigned #${v.number as number}`;
  if (c.scope === 'identification') return 'Person grouping and thresholds';
  if (c.scope === 'roster') return 'Roster';
  return c.id;
}

/** The label edit a candidate can become, if any. */
function operationOf(c: Candidate): Operation | null {
  if (c.can_remove_event) return 'remove_event';
  if (!c.rally) return null;
  if (c.scope === 'rally') return 'remove_rally';
  if (c.scope === 'rally_bounds') return 'update_rally';
  return null;
}

/** Operator-published matches stream a pipeline cut; that cut is the label target. */
function sourceCut(match: LibraryMatch): string {
  const key = match.source_video?.public_url ? null : match.source_video?.r2_key;
  return key?.split('/').at(-1) ?? '';
}

/** Candidates by rally, each group in review order; match-wide ones last. */
function groups(candidates: Candidate[]): [number | null, Candidate[]][] {
  const out = new Map<number | null, Candidate[]>();
  const sorted = [...candidates].sort(byReviewOrder);
  for (const c of sorted) out.set(c.rally_index, [...(out.get(c.rally_index) ?? []), c]);
  return [...out].sort(([a], [b]) => (a ?? Infinity) - (b ?? Infinity));
}

export function FeedbackPanel({
  user,
  detail,
  selected,
  onSelect,
}: {
  user: string;
  detail: MatchDetail;
  /** The open correction's id. */
  selected: string;
  onSelect: (id: string) => void;
}) {
  const queryClient = useQueryClient();
  const [reviewId, setReviewId] = useState(detail.review_id);
  const [video, setVideo] = useState(() => sourceCut(detail.match));
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
  const grouped = groups(candidates);
  const ordered = grouped.flatMap(([, entries]) => entries);
  const candidate = candidates.find((c) => c.id === selected);
  const decisions = review.data?.decisions ?? {};
  const loaded = !reviewId || !!review.data;

  // Open on the first correction still to review (also when the URL names
  // one this snapshot lacks).
  const first = loaded ? (ordered.find((c) => !decisions[c.id]) ?? ordered[0]) : undefined;
  useEffect(() => {
    if (!candidate && first) onSelect(first.id);
  }, [candidate, first, onSelect]);
  // Keep the open correction in view as it moves (Edited chip, auto-advance).
  useEffect(() => {
    if (selected) document.querySelector(`[data-action-id="${CSS.escape(selected)}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [selected]);

  const run = async (fn: () => Promise<Review>, done: string) => {
    setBusy(true);
    try {
      const next = await fn();
      queryClient.setQueryData(['app-review', 'review', next.id], next);
      setReviewId(next.id);
      void queryClient.invalidateQueries({ queryKey: ['app-review', 'match'] });
      void target.refetch();
      toast.success(done);
      return next;
    } catch (e) {
      toast.error(errMsg(e));
      return null;
    } finally {
      setBusy(false);
    }
  };
  const decide = async (decision: string, note: string, sameSource: boolean) => {
    const at = ordered.findIndex((c) => c.id === candidate!.id);
    const next = await run(async () => {
      // The first verdict pins the snapshot it judges.
      const id =
        reviewId ??
        (await apiFetch<Review>(API.appReview.startReview(user, detail.match.id), { method: 'POST' })).id;
      return apiFetch<Review>(API.appReview.decisions(id), {
        method: 'POST',
        body: {
          candidate_id: candidate!.id,
          decision,
          note,
          video,
          revision: target.data?.[decision] || '',
          same_source: sameSource,
        },
      });
    }, DECISION_LABEL[decision] ?? decision);
    // Move on to the next correction still to review, wrapping around.
    const after = next && [...ordered.slice(at + 1), ...ordered.slice(0, at)].find((c) => !next.decisions[c.id]);
    if (after) onSelect(after.id);
  };

  if (candidates.length === 0)
    return <p className="py-6 text-center text-xs text-text-muted">This match has no user corrections.</p>;
  if (review.error) return <ErrorState message={errMsg(review.error)} onRetry={() => void review.refetch()} />;

  return (
    <div className="flex min-h-0 flex-col gap-3 text-sm">
      <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
        Reviewed {Object.keys(decisions).length}/{candidates.length}
        {detail.reviews.length > 1 && (
          <select
            aria-label="Review snapshot"
            className={cn(fieldCls, 'h-8 w-auto py-0 text-xs')}
            value={reviewId ?? ''}
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
        {!reviewId && (
          <span className="basis-full text-[11px]">
            The first verdict pins this App snapshot — later edits by the user do not change it.
          </span>
        )}
      </div>
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

      {candidate ? (
        <Verdict
          key={candidate.id}
          candidate={candidate}
          previous={decisions[candidate.id]}
          locked={busy || !loaded || review.data?.application != null}
          video={video}
          onVideo={setVideo}
          videos={videos}
          target={target}
          onDecide={(d, note, same) => void decide(d, note, same)}
        />
      ) : (
        <p className="rounded-xl border border-dashed border-border px-3 py-4 text-center text-xs text-text-muted">
          Loading the review…
        </p>
      )}

      <div className="vq-list max-h-[calc(100vh-24rem)] min-h-0 space-y-3 overflow-y-auto pr-1 [&_.ml-6]:ml-0">
        {grouped.map(([rally, entries]) => (
          <div key={rally ?? 'match'} className="space-y-1.5">
            <p className="px-1 text-[11px] font-medium text-text-secondary">
              {rally == null ? 'Whole match' : `Rally #${rally}`}
            </p>
            <EventRows
              entries={entries}
              empty="No corrections"
              columns="minmax(0,1fr) auto"
              selectedId={selected || null}
              isActive={() => false}
              onJump={(c) => onSelect(c.id)}
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
        ))}
      </div>
    </div>
  );
}

/** The open correction: what it is, the verdict, and — where the change can
 *  become a label edit — the evidenced write-back. Keyed by candidate, so
 *  the note and the confirmation start empty for each one. */
function Verdict({
  candidate,
  previous,
  locked,
  video,
  onVideo,
  videos,
  target,
  onDecide,
}: {
  candidate: Candidate;
  previous: Decision | undefined;
  locked: boolean;
  video: string;
  onVideo: (video: string) => void;
  videos: UseQueryResult<{ name: string }[]>;
  target: UseQueryResult<Record<string, string | null>>;
  onDecide: (decision: string, note: string, sameSource: boolean) => void;
}) {
  const [note, setNote] = useState('');
  const [sameSource, setSameSource] = useState(false);
  const operation = operationOf(candidate);
  const written = ['remove_event', 'remove_rally', 'update_rally'].includes(previous?.decision ?? '');
  const disabled = locked || written;

  return (
    <div className="space-y-3 rounded-xl border border-primary/30 bg-primary/[0.04] p-3">
      <div>
        <p className="text-xs font-medium text-text-primary">
          {candidate.rally_index != null && `Rally #${candidate.rally_index} · `}
          {title(candidate)}
        </p>
        {candidate.reason && <p className="mt-1 text-xs text-amber-400">{candidate.reason}</p>}
        {previous && (
          <p className="mt-1 text-xs text-text-muted">
            {DECISION_LABEL[previous.decision] ?? previous.decision} · {previous.actor} · {previous.at}
            {previous.note && <span className="block text-text-secondary">{previous.note}</span>}
          </p>
        )}
      </div>
      <textarea
        aria-label="Review note"
        className={cn(fieldCls, 'text-xs')}
        rows={2}
        placeholder="What the video shows, why it is a misdetection, or why the change is not adopted"
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />
      <div className="flex flex-wrap gap-2">
        <Button
          size="sm"
          disabled={disabled}
          onClick={() => onDecide('accepted_feedback', note, false)}
          title="Agree with the user; keep it as training feedback without editing a label"
        >
          Adopt as feedback
        </Button>
        <Button size="sm" disabled={disabled} onClick={() => onDecide('rejected', note, false)}>
          Don't adopt
        </Button>
      </div>
      {operation && (
        <Collapsible label="Write back to the label">
          <div className="space-y-2">
            <VideoCombobox
              items={videos.data ?? []}
              value={video}
              onChange={onVideo}
              query={videos}
              placeholder="Pipeline video to write…"
            />
            <label className="flex gap-2 text-xs text-text-secondary">
              <input type="checkbox" checked={sameSource} onChange={(e) => setSameSource(e.target.checked)} />
              Same source and timeline as the App video, and the change follows the label definition — not a
              viewing preference.
            </label>
            <Button
              size="sm"
              intent="danger"
              disabled={disabled || !sameSource || !note.trim() || !target.data?.[operation]}
              title={!note.trim() ? 'Write a note first' : undefined}
              onClick={() => onDecide(operation, note, sameSource)}
            >
              {WRITE_BACK[operation]}
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
        </Collapsible>
      )}
      <Collapsible label="Raw correction">
        <pre className="max-h-48 overflow-auto rounded bg-surface-100 p-2 text-xs">
          {JSON.stringify(candidate.value, null, 2)}
        </pre>
      </Collapsible>
    </div>
  );
}

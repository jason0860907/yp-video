/** App Review: browse any VolleyIQ user's library the way the App shows it,
 *  and review their corrections as training feedback.
 *
 *  A user without a picked match shows the Overview: every match with its
 *  jobs and edit counts. A match has two modes, as tabs: Clips shows it
 *  through the App's feeds and framing; Review lists the user's corrections
 *  against the original analysis, so what they deleted is still on the
 *  timeline. A header card holds the pickers and a one-line match summary;
 *  where you are lives in the URL (useAppReviewUrlState) so links are
 *  shareable — the admin dashboard links straight to a match.
 */

import { useCallback, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { ErrorState } from '@/components/ui/ErrorState';
import { Segmented } from '@/components/ui/Segmented';
import { Tabs } from '@/components/ui/Tabs';
import { ClipList } from './ClipList';
import { ClipPlayer, type ClipPlayerHandle } from './ClipPlayer';
import { FeedbackPanel } from './FeedbackPanel';
import { JobRow } from './JobStatus';
import { LibraryPicker } from './LibraryPicker';
import { MatchSummary } from './MatchSummary';
import { Overview } from './Overview';
import { useAppReviewUrlState } from './useAppReviewUrlState';
import type { Candidate, Clip, Library, MatchDetail, VideoMeta, WindowMode } from './types';
import { byReviewOrder, liveMatches } from './types';

const FRAMINGS: { key: WindowMode; label: string }[] = [
  { key: 'full_play', label: 'Full play' },
  { key: 'to_next', label: 'Single touch' },
  { key: 'whole_rally', label: 'Whole rally' },
];

export function AppReviewPage() {
  const { user, match, tab, scope, clip, fix, set } = useAppReviewUrlState();
  const [mode, setMode] = useState<WindowMode>('full_play');
  const [corrected, setCorrected] = useState(true);
  // Review judges the edits against the original analysis.
  const applied = tab === 'clips' && corrected;
  const queue = useRef<Clip[]>([]);
  const player = useRef<ClipPlayerHandle>(null);

  const library = useQuery({
    queryKey: ['app-review', 'library', user],
    enabled: !!user,
    queryFn: () => apiFetch<Library>(API.appReview.library(user)),
  });
  const picked = library.data?.matches.find((m) => m.id === match);
  // Until its analysis has a result there is nothing to project: the match
  // shows its jobs instead.
  const analysed = !!picked?.r2_keys.result;
  const detail = useQuery({
    queryKey: ['app-review', 'match', user, match, mode, applied],
    enabled: !!user && !!match && analysed,
    // Reframing (mode / corrections) keeps the old clips on screen meanwhile.
    placeholderData: (prev, query) => (query?.queryKey[3] === match ? prev : undefined),
    queryFn: () => apiFetch<MatchDetail>(API.appReview.match(user, match, mode, applied)),
  });
  const meta = useQuery({
    queryKey: ['app-review', 'video-meta', user, match],
    enabled: !!user && !!match && analysed,
    staleTime: Infinity,
    queryFn: () => apiFetch<VideoMeta>(API.appReview.videoMeta(user, match)),
  });

  const rallyTags = useMemo(() => {
    const out = new Map<string, string[]>();
    for (const rt of library.data?.rally_tags ?? []) {
      if (rt.match_id !== match || rt.deleted_at !== null) continue;
      const key = rt.rally_id.toLowerCase();
      out.set(key, [...(out.get(key) ?? []), rt.tag_id]);
    }
    return out;
  }, [library.data, match]);

  const candidates = detail.data?.candidates;
  /** Rally index → the user's corrections inside it. */
  const edits = useMemo(() => {
    const out = new Map<number, Candidate[]>();
    for (const c of [...(candidates ?? [])].sort(byReviewOrder))
      if (c.rally_index != null) out.set(c.rally_index, [...(out.get(c.rally_index) ?? []), c]);
    return out;
  }, [candidates]);
  // The open correction's span, played without being one of the feed's clips.
  const fixed = candidates?.find((c) => c.id === fix);
  const fixClip = useMemo<Clip | null>(
    () =>
      fixed?.clip
        ? {
            key: `fix:${fixed.id}`,
            kind: fixed.clip.kind ?? (fixed.rally_index != null ? 'rally' : 'other'),
            rally_index: fixed.rally_index,
            touches: fixed.clip.touches ?? [],
            start: fixed.clip.start,
            end: fixed.clip.end,
          }
        : null,
    [fixed],
  );

  const preview = detail.data?.preview;
  const feed = preview?.[scope];
  // Clips opens on the feed's first clip, so a picked match never sits blank.
  const active = tab === 'review' ? fixClip : (feed?.find((c) => c.key === clip) ?? feed?.[0] ?? null);
  const select = (next: Clip, visible: Clip[]) => {
    queue.current = visible;
    set({ clip: next.key });
  };
  const playNext = useCallback(() => {
    if (tab !== 'clips' || !active) return;
    const list = queue.current.length ? queue.current : (feed ?? []);
    const i = list.findIndex((c) => c.key === active.key);
    const next = list[i + 1];
    if (i >= 0 && next) set({ clip: next.key });
  }, [tab, active, feed, set]);

  const error = detail.error ?? meta.error ?? library.error;
  const ready = detail.data && preview && meta.data;

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <Card>
        {match && (
          <Tabs
            tabs={[
              { key: 'clips', label: 'Clips' },
              { key: 'review', label: `Review${candidates ? ` (${candidates.length})` : ''}` },
            ]}
            active={tab}
            onChange={(t) => set({ tab: t })}
          />
        )}
        <div className={cn('flex flex-wrap items-center gap-x-5 gap-y-2', match && 'mt-3')}>
          <LibraryPicker
            user={user}
            match={match}
            library={library.data}
            onUser={(id) => set({ user: id, match: '', clip: '', fix: '' })}
            onMatch={(id) => set({ match: id, clip: '', fix: '' })}
          />
          {match && (
            <Button size="sm" intent="ghost" onClick={() => set({ match: '', clip: '', fix: '' })}>
              ← All matches
            </Button>
          )}
          {tab === 'clips' && match && (
            <>
              <span className="inline-flex items-center gap-2 text-xs text-text-muted">
                Framing
                <Segmented options={FRAMINGS} value={mode} onChange={setMode} />
              </span>
              <label className="inline-flex items-center gap-1.5 text-xs text-text-secondary">
                <input type="checkbox" checked={corrected} onChange={(e) => setCorrected(e.target.checked)} />
                Apply user corrections
              </label>
            </>
          )}
        </div>
        {detail.data && <MatchSummary match={detail.data.match} preview={detail.data.preview} />}
      </Card>

      {!user ? (
        <Card>
          <EmptyState
            icon={<PlayIcon />}
            title="Pick a user"
            subtitle="See their matches, clips and corrections the way the App shows them"
          />
        </Card>
      ) : !match ? (
        <Card>
          {library.error ? (
            <ErrorState message={errMsg(library.error)} onRetry={() => void library.refetch()} />
          ) : library.data ? (
            <Overview user={user} matches={liveMatches(library.data)} onMatch={(id) => set({ match: id, clip: '', fix: '' })} />
          ) : (
            <p className="py-12 text-center text-xs text-text-muted">Loading library…</p>
          )}
        </Card>
      ) : picked && !analysed ? (
        <Card>
          <div className="space-y-3 py-6 text-center text-xs text-text-muted">
            <p className="text-sm text-text-secondary">{picked.title} has no analysis result yet</p>
            <div className="flex justify-center">
              <JobRow match={picked} />
            </div>
          </div>
        </Card>
      ) : error ? (
        <Card>
          <ErrorState
            message={errMsg(error)}
            onRetry={() => {
              void detail.refetch();
              void meta.refetch();
            }}
          />
        </Card>
      ) : !ready ? (
        <Card>
          <p className="py-12 text-center text-xs text-text-muted">Loading match…</p>
        </Card>
      ) : (
        <div className="flex flex-col gap-5 xl:flex-row">
          <ClipPlayer
            ref={player}
            title={detail.data.match.title}
            src={apiUrl(API.appReview.video(user, match))}
            meta={meta.data}
            clip={active}
            rallies={preview.rallies}
            onEnded={playNext}
          />
          <div className="min-w-0 xl:w-[460px] xl:flex-shrink-0">
            <Card>
              {tab === 'clips' ? (
                <ClipList
                  preview={preview}
                  tags={library.data?.tags ?? []}
                  rallyTags={rallyTags}
                  edits={edits}
                  scope={scope}
                  selected={active}
                  onScope={(s) => {
                    queue.current = [];
                    set({ scope: s, clip: '' });
                  }}
                  onSelect={select}
                  onSeek={(t) => player.current?.seek(t)}
                  onOpenEdit={(c) => set({ tab: 'review', fix: c.id })}
                />
              ) : (
                <FeedbackPanel
                  key={`${match}:${detail.data.review_id}`}
                  user={user}
                  detail={detail.data}
                  selected={fix}
                  onSelect={(id) => set({ fix: id })}
                />
              )}
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

function PlayIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 010 1.972l-11.54 6.347a1.125 1.125 0 01-1.667-.986V5.653z" />
    </svg>
  );
}

/** App Review: browse any VolleyIQ user's library the way the App shows it,
 *  and review their corrections as training feedback.
 *
 *  Laid out like Label: a header card (tabs, then the pickers and the App's
 *  framing settings), the player, and a sidebar the tab fills — the App's
 *  feeds, match info, or the correction review. Where you are lives in the
 *  URL (useAppReviewUrlState) so links are shareable.
 */

import { useCallback, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { ErrorState } from '@/components/ui/ErrorState';
import { Segmented } from '@/components/ui/Segmented';
import { Tabs } from '@/components/ui/Tabs';
import { ClipList } from './ClipList';
import { ClipPlayer, type ClipPlayerHandle } from './ClipPlayer';
import { FeedbackPanel } from './FeedbackPanel';
import { LibraryPicker } from './LibraryPicker';
import { MatchInfo } from './MatchInfo';
import { useAppReviewUrlState } from './useAppReviewUrlState';
import type { Clip, Library, MatchDetail, VideoMeta, WindowMode } from './types';

const FRAMINGS: { key: WindowMode; label: string }[] = [
  { key: 'full_play', label: 'Full play' },
  { key: 'to_next', label: 'Single touch' },
  { key: 'whole_rally', label: 'Whole rally' },
];

export function AppReviewPage() {
  const { user, match, tab, scope, clip, set } = useAppReviewUrlState();
  const [mode, setMode] = useState<WindowMode>('full_play');
  const [corrected, setCorrected] = useState(true);
  // A review candidate's span, played without being one of the feed's clips.
  const [focus, setFocus] = useState<Clip | null>(null);
  const queue = useRef<Clip[]>([]);
  const player = useRef<ClipPlayerHandle>(null);

  const library = useQuery({
    queryKey: ['app-review', 'library', user],
    enabled: !!user,
    queryFn: () => apiFetch<Library>(API.appReview.library(user)),
  });
  const detail = useQuery({
    queryKey: ['app-review', 'match', user, match, mode, corrected],
    enabled: !!user && !!match,
    // Reframing (mode / corrections) keeps the old clips on screen meanwhile.
    placeholderData: (prev, query) => (query?.queryKey[3] === match ? prev : undefined),
    queryFn: () => apiFetch<MatchDetail>(API.appReview.match(user, match, mode, corrected)),
  });
  const meta = useQuery({
    queryKey: ['app-review', 'video-meta', user, match],
    enabled: !!user && !!match,
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

  const preview = detail.data?.preview;
  const active = focus ?? preview?.[scope].find((c) => c.key === clip) ?? null;
  const select = (next: Clip, visible: Clip[]) => {
    queue.current = visible;
    setFocus(null);
    set({ clip: next.key });
  };
  const playNext = useCallback(() => {
    const i = queue.current.findIndex((c) => c.key === clip);
    const next = queue.current[i + 1];
    if (i >= 0 && next) set({ clip: next.key });
  }, [clip, set]);

  const error = detail.error ?? meta.error ?? library.error;
  const ready = detail.data && preview && meta.data;

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <Card>
        <Tabs
          tabs={[
            { key: 'clips', label: 'Clips' },
            { key: 'match', label: 'Match' },
            { key: 'review', label: `Review${detail.data ? ` (${detail.data.candidates.length})` : ''}` },
          ]}
          active={tab}
          onChange={(t) => set({ tab: t })}
        />
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2">
          <LibraryPicker
            user={user}
            match={match}
            library={library.data}
            onUser={(id) => {
              setFocus(null);
              set({ user: id, match: '', clip: '' });
            }}
            onMatch={(id) => {
              setFocus(null);
              set({ match: id, clip: '' });
            }}
          />
          <span className="inline-flex items-center gap-2 text-xs text-text-muted">
            Framing
            <Segmented options={FRAMINGS} value={mode} onChange={setMode} />
          </span>
          <label className="inline-flex items-center gap-1.5 text-xs text-text-secondary">
            <input type="checkbox" checked={corrected} onChange={(e) => setCorrected(e.target.checked)} />
            Apply user corrections
          </label>
        </div>
      </Card>

      {!user || !match ? (
        <Card>
          <EmptyState
            icon={<PlayIcon />}
            title={user ? 'Pick a match' : 'Pick a user'}
            subtitle="See their matches, clips and corrections the way the App shows them"
          />
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
              {tab === 'clips' && (
                <ClipList
                  preview={preview}
                  tags={library.data?.tags ?? []}
                  rallyTags={rallyTags}
                  scope={scope}
                  selected={focus ? null : active}
                  onScope={(s) => set({ scope: s, clip: '' })}
                  onSelect={select}
                  onSeek={(t) => player.current?.seek(t)}
                />
              )}
              {tab === 'match' && <MatchInfo match={detail.data.match} preview={preview} />}
              {tab === 'review' && (
                <FeedbackPanel
                  key={`${match}:${detail.data.review_id}`}
                  user={user}
                  detail={detail.data}
                  onSeek={(span) =>
                    setFocus({
                      key: `focus:${span.start}:${span.end}`,
                      kind: 'other',
                      rally_index: null,
                      touches: [],
                      ...span,
                    })
                  }
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

/** App Review: browse any VolleyIQ user's library the way the App shows it,
 *  and review their corrections as training feedback.
 *
 *  Left: users → matches. Center: the source video, seeking between clips.
 *  Right: the App's feeds, match info, and the correction review. Where you
 *  are (?user=&match=&scope=&clip=) lives in the URL so links are shareable.
 */

import { useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
import { Card } from '@/components/ui/Card';
import { ClipList } from './ClipList';
import { ClipPlayer } from './ClipPlayer';
import { FeedbackPanel } from './FeedbackPanel';
import { LibraryBrowser } from './LibraryBrowser';
import { MatchInfo } from './MatchInfo';
import type { Clip, Library, MatchDetail, Scope, WindowMode } from './types';

type Tab = 'clips' | 'info' | 'review';
const SCOPES: readonly string[] = ['rallies', 'actions', 'scores'];

function useUrlState() {
  const [params, setParams] = useSearchParams();
  const raw = params.get('scope');
  const state = {
    user: params.get('user') ?? '',
    match: params.get('match') ?? '',
    scope: (raw && SCOPES.includes(raw) ? raw : 'rallies') as Scope,
    clip: params.get('clip') ?? '',
  };
  const set = (next: Partial<typeof state>) =>
    setParams(
      (prev) => {
        const out = new URLSearchParams(prev);
        for (const [k, v] of Object.entries(next)) {
          if (v) out.set(k, v);
          else out.delete(k);
        }
        return out;
      },
      { replace: true },
    );
  return [state, set] as const;
}

export function AppReviewPage() {
  const [{ user, match, scope, clip }, set] = useUrlState();
  const [tab, setTab] = useState<Tab>('clips');
  const [mode, setMode] = useState<WindowMode>('full_play');
  const [corrected, setCorrected] = useState(true);
  // A review candidate's span, played without being one of the feed's clips.
  const [focus, setFocus] = useState<Clip | null>(null);
  const queue = useRef<Clip[]>([]);

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
  const playNext = () => {
    const i = queue.current.findIndex((c) => c.key === active?.key);
    const next = queue.current[i + 1];
    if (i >= 0 && next) set({ clip: next.key });
  };

  return (
    <div className="mx-auto grid max-w-[1800px] gap-4 xl:grid-cols-[260px_minmax(0,1fr)_400px]">
      <Card className="xl:max-h-[calc(100vh-120px)] xl:overflow-auto">
        <LibraryBrowser
          user={user}
          match={match}
          onUser={(id) => set({ user: id, match: '', clip: '' })}
          onMatch={(id) => {
            setFocus(null);
            set({ match: id, clip: '' });
          }}
        />
      </Card>

      <div className="min-w-0 space-y-3">
        {!match ? (
          <Card>
            <p className="text-sm text-text-muted">
              {user ? '選一場比賽。' : '選一位用戶，看他在 App 裡的比賽、片段與修正。'}
            </p>
          </Card>
        ) : (
          <Card
            label={detail.data?.match.title ?? '載入中…'}
            right={
              <div className="flex items-center gap-3">
                <select
                  aria-label="取景"
                  className={cn(fieldCls, 'h-8 w-auto py-0 text-xs')}
                  value={mode}
                  onChange={(e) => setMode(e.target.value as WindowMode)}
                >
                  <option value="full_play">組織進攻</option>
                  <option value="to_next">單一動作</option>
                  <option value="whole_rally">整個 Rally</option>
                </select>
                <label className="flex items-center gap-1.5 text-xs text-text-secondary">
                  <input
                    type="checkbox"
                    checked={corrected}
                    onChange={(e) => setCorrected(e.target.checked)}
                  />
                  套用使用者修正
                </label>
              </div>
            }
          >
            {detail.error ? (
              <p role="alert" className="text-sm text-red-400">
                {errMsg(detail.error)}
              </p>
            ) : (
              <ClipPlayer
                src={apiUrl(API.appReview.video(user, match))}
                clip={active}
                onEnded={playNext}
              />
            )}
          </Card>
        )}
      </div>

      {detail.data && preview && (
        <Card className="flex flex-col xl:max-h-[calc(100vh-120px)]">
          <div className="-mt-1 mb-3 flex gap-1 border-b border-border">
            {(
              [
                ['clips', '片段'],
                ['info', '比賽資訊'],
                ['review', `修正審核 ${detail.data.candidates.length}`],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={cn(
                  '-mb-px border-b-2 px-3 pb-2 text-xs font-medium',
                  key === tab
                    ? 'border-primary text-text-primary'
                    : 'border-transparent text-text-secondary hover:text-text-primary',
                )}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex min-h-0 flex-1 flex-col overflow-auto">
            {tab === 'clips' && (
              <ClipList
                preview={preview}
                tags={library.data?.tags ?? []}
                rallyTags={rallyTags}
                scope={scope}
                selected={focus ? '' : clip}
                onScope={(s) => set({ scope: s, clip: '' })}
                onSelect={select}
              />
            )}
            {tab === 'info' && <MatchInfo match={detail.data.match} preview={preview} />}
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
          </div>
        </Card>
      )}
    </div>
  );
}

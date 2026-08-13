/** Unified Label page: pick the video once, switch labeling modes via tabs.
 *
 *  One picker over the client-side union of the four work lists (see
 *  useUnionVideos); each mode's Status semantics come from its panel's MODE
 *  descriptor, so the select here filters with rules that live beside the
 *  panel they describe. State is in the URL (?video=&mode=) for shareable,
 *  refresh-safe links.
 *
 *  Only the active panel is mounted — react-query's shared keys make
 *  Association↔ReID switches cheap and Action's localStorage drafts survive
 *  unmount. Panels with unsaved work register a dirty guard; every video or
 *  mode change awaits it before touching the URL.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { fieldCls } from '@/components/form/Field';
import { apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import type { LabelMode } from '@/lib/labelStatus';
import { useUnionVideos } from '@/lib/useUnionVideos';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { toast } from '@/components/feedback/toast';
import { CopyFilenameButton } from '@/components/video/CopyFilenameButton';
import { KindBadge } from '@/components/video/KindBadge';
import { PipelineChips } from '@/components/video/PipelineChips';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { RALLY_MODE, RallyPanel } from './RallyPanel';
import { ACTION_MODE, ActionPanel } from './ActionPanel';
import { ASSOCIATION_MODE, AssociationPanel } from './AssociationPanel';
import { REID_MODE, ReidPanel } from './ReidPanel';
import { SourceSelect } from './SourceSelect';
import { StatusChip } from './StatusChip';
import { useLabelUrlState } from './useLabelUrlState';
import type {
  DirtyGuard,
  LabelSource,
  LoadedSource,
  ModeDescriptor,
  PlaybackClock,
} from './mode';

const MODES: ModeDescriptor[] = [RALLY_MODE, ACTION_MODE, ASSOCIATION_MODE, REID_MODE];

type KindFilter = 'all' | 'broadcast' | 'sideline';

export function LabelPage() {
  const { video, mode, set } = useLabelUrlState();
  const { videos, query } = useUnionVideos();
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [statusFilter, setStatusFilter] = useState('all');
  // The "which store to read" choice — a per-load setting, not a list
  // filter, so it renders beside the mode tabs instead of in the picker row.
  // Shared between rally and action (the two multi-store modes); the VLM
  // checkbox is rally's third store, exposed only there.
  const [source, setSource] = useState<LabelSource>('annotation');
  const [vlm, setVlm] = useState(false);
  // What the panel's last load actually resolved to — 'none' means the
  // selected store has no file for this video yet (an empty editor).
  const [loadedSource, setLoadedSource] = useState<LoadedSource | null>(null);
  // Saving always writes the human store; if the user was viewing the
  // machine store, follow the save there so the screen shows what's saved.
  const onPanelSaved = () => setSource((s) => (s === 'pre-annotation' ? 'annotation' : s));
  useEffect(() => setLoadedSource(null), [video, mode, source, vlm]);

  const active = MODES.find((m) => m.key === mode) ?? RALLY_MODE;
  // A status the current mode doesn't offer (left over from another tab)
  // falls back to 'all' rather than silently filtering by stale semantics.
  const status = active.statusOptions.some((o) => o.value === statusFilter) ? statusFilter : 'all';

  const visible = useMemo(
    () =>
      videos.filter(
        (v) => (kindFilter === 'all' || v.kind === kindFilter) && active.matches(v, status),
      ),
    [videos, kindFilter, active, status],
  );
  // Undefined while the lists load — the URL's video is trusted as-is so a
  // shared link starts loading its panel before (or without) a listing row.
  const pickedRow = useMemo(() => videos.find((v) => v.name === video), [videos, video]);

  // The active panel's unsaved-work gate. Only panels that can hold unsaved
  // state register one (Action, ReID); the ref is null otherwise.
  const guardRef = useRef<DirtyGuard | null>(null);
  const registerGuard = useCallback((g: DirtyGuard | null) => {
    guardRef.current = g;
  }, []);

  // Shared playhead: Rally and Action write while playing, read on load, so
  // a tab switch resumes at the same time. A ref, not state — position moves
  // several times a second and nothing here needs to re-render for it.
  const clockStore = useRef<{ video: string; t: number } | null>(null);
  const clock = useMemo<PlaybackClock>(
    () => ({
      read: (v) => (clockStore.current?.video === v ? clockStore.current.t : null),
      write: (v, t) => {
        clockStore.current = { video: v, t };
      },
    }),
    [],
  );
  const guarded = async (apply: () => void) => {
    if (guardRef.current && !(await guardRef.current())) return;
    apply();
  };

  const pickVideo = (name: string) => {
    if (name === video) return;
    void guarded(() => set({ video: name }));
  };
  const pickMode = (m: LabelMode) => {
    if (m === mode) return;
    void guarded(() => set({ mode: m }));
  };

  const unavailable = pickedRow && !active.available(pickedRow);

  // Page-level Done toggle for modes whose flag is a plain stored bit.
  // ReID has no doneApi — its panel button saves the board first and can
  // confirm auto actors, so the page defers to it.
  const qc = useQueryClient();
  const [doneBusy, setDoneBusy] = useState(false);
  const isDone = pickedRow ? active.status(pickedRow) === 'done' : false;
  const toggleDone = async () => {
    if (!video || !active.doneApi) return;
    setDoneBusy(true);
    try {
      // Association's Done also sweep-confirms current auto picks; the
      // response says how many landed (0 for the other modes).
      const res = await apiFetch<{ done: boolean; confirmed?: number }>(active.doneApi(video), {
        method: 'PUT',
        body: { done: !isDone },
      });
      toast.success(
        isDone
          ? 'Done mark removed'
          : res.confirmed
            ? `${active.label} labeling marked done — ${res.confirmed} auto pick(s) confirmed`
            : `${active.label} labeling marked done`,
      );
      void qc.invalidateQueries({ queryKey: [active.listKey] });
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
      if (res.confirmed) void qc.invalidateQueries({ queryKey: ['extraction-records', video] });
    } catch (e) {
      toast.error(`Done failed: ${errMsg(e)}`);
    } finally {
      setDoneBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <Card>
        {/* Tier 1 — where you are. Underline tabs sit on the divider so the
            header reads as one line, not a box floating among form controls.
            Tabs are disabled where the picked video has nothing to open. */}
        <div className="flex flex-wrap items-center gap-1 border-b border-border">
          {MODES.map((m) => {
            const disabled = Boolean(pickedRow && !m.available(pickedRow));
            return (
              <button
                key={m.key}
                type="button"
                disabled={disabled}
                onClick={() => pickMode(m.key)}
                title={disabled && pickedRow ? m.hint(pickedRow) : undefined}
                className={cn(
                  '-mb-px border-b-2 px-4 pb-2 pt-1 text-xs font-medium transition-colors',
                  m.key === mode
                    ? 'border-primary text-text-primary'
                    : disabled
                      ? // A muted rose, color alone: distinct from the idle
                        // gray without shouting like a full warning tint.
                        'cursor-not-allowed border-transparent text-red-400/45'
                      : 'border-transparent text-text-secondary hover:border-border-light hover:text-text-primary',
                )}
              >
                {m.label}
              </button>
            );
          })}
        </div>

        {/* Tier 2 — settings: what slice of the library, which store. */}
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2">
          <label className="inline-flex items-center gap-2 text-xs text-text-muted">
            Kind
            <select
              value={kindFilter}
              onChange={(e) => setKindFilter(e.target.value as KindFilter)}
              className={cn(fieldCls, 'h-9 py-0')}
            >
              <option value="all">All kinds</option>
              <option value="broadcast">Broadcast</option>
              <option value="sideline">Sideline</option>
            </select>
          </label>
          <label className="inline-flex items-center gap-2 text-xs text-text-muted">
            Status
            <select
              value={status}
              onChange={(e) => setStatusFilter(e.target.value)}
              className={cn(fieldCls, 'h-9 py-0')}
            >
              {active.statusOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          {active.hasSources && (
            <SourceSelect
              source={source}
              onSource={setSource}
              vlm={vlm}
              onVlm={setVlm}
              showVlm={active.key === 'rally'}
              loaded={loadedSource}
            />
          )}
          {/* The picker takes whatever width is left; on narrow screens the
              whole group (search + copy + done) wraps to its own line. */}
          <div className="flex min-w-[20rem] flex-1 items-center gap-2">
            <VideoCombobox
              className="min-w-0 flex-1"
              items={visible}
              value={video}
              onChange={pickVideo}
              query={query}
              placeholder={`Search ${visible.length} videos…`}
              renderItem={(v) => {
                // Each row speaks only the active mode's language: its status
                // chips when this tab can open the video, otherwise a dimmed
                // name plus what's blocking it (pipeline chips when we have
                // them, hover for the full hint).
                const ready = active.available(v);
                const pipeline = (v.reid ?? v.assoc)?.pipeline;
                return (
                  <>
                    <KindBadge kind={v.kind} />
                    <span
                      className={cn('min-w-0 flex-1 break-all font-mono', !ready && 'opacity-50')}
                    >
                      {v.name}
                    </span>
                    {ready ? (
                      <>
                        {active.rowExtras?.(v)}
                        <StatusChip status={active.status(v)} />
                      </>
                    ) : (
                      <span title={active.hint(v)} className="inline-flex shrink-0 items-center">
                        {pipeline ? (
                          <PipelineChips pipeline={pipeline} />
                        ) : (
                          <Badge tone="warning">not ready</Badge>
                        )}
                      </span>
                    )}
                  </>
                );
              }}
            />
            <CopyFilenameButton name={video} />
            {active.doneApi && (
              <Button
                size="sm"
                intent={isDone ? 'default' : 'primary'}
                onClick={() => void toggleDone()}
                disabled={!video || doneBusy || Boolean(unavailable)}
                title={
                  isDone
                    ? 'Labeling marked finished — click to unmark'
                    : `Mark this video's ${active.label} labeling as finished`
                }
              >
                {isDone ? 'Done ✓' : 'Done'}
              </Button>
            )}
          </div>
        </div>
      </Card>

      {!video && query.isPending ? (
        // Loading, not "empty": until the work lists arrive, "Pick a video"
        // over a zero-row picker reads as "there are no videos".
        <Card>
          <p className="py-12 text-center text-xs text-text-muted">Loading videos…</p>
        </Card>
      ) : !video ? (
        <Card>
          <EmptyState
            icon={
              <svg
                className="h-5 w-5"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                />
              </svg>
            }
            title="Pick a video"
            subtitle="One pick works across all four labeling modes"
          />
        </Card>
      ) : unavailable ? (
        <Card>
          <p className="text-xs text-amber-400">
            {active.label} labeling is not available for this video — {active.hint(pickedRow)}.
          </p>
        </Card>
      ) : mode === 'rally' ? (
        <RallyPanel
          video={video}
          source={source}
          vlm={vlm}
          onLoaded={setLoadedSource}
          onSaved={onPanelSaved}
          clock={clock}
        />
      ) : mode === 'action' ? (
        <ActionPanel
          video={video}
          source={source}
          onLoaded={setLoadedSource}
          onSaved={onPanelSaved}
          registerGuard={registerGuard}
          clock={clock}
        />
      ) : mode === 'association' ? (
        <AssociationPanel video={video} />
      ) : (
        <ReidPanel video={video} registerGuard={registerGuard} />
      )}
    </div>
  );
}

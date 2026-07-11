import { useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { CropImage } from '@/components/video/CropImage';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { ReidCluster, ReidPlayers, ReidRecord, ReidVideo } from '@/types/api';

// CLIP-ReID's ViT features sit in a tight cosine cone, hence the small values.
const THRESHOLDS = [0.12, 0.15, 0.18, 0.21, 0.24];
const DEFAULT_THRESHOLD = 0.15;
// Below this cosine similarity a predicted identity is rendered as doubtful.
const LOW_SIM = 0.6;

const STATUS_DOT: Record<ReidRecord['status'], string> = {
  ok: 'bg-primary-light',
  multi: 'bg-amber-400',
  miss: 'bg-red-400',
};

const selectCls =
  'w-auto cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-1 text-xs text-text-primary focus:border-primary/50 focus:outline-none';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

/** One editable identity group: named = a player, unnamed = an auto cluster.
 *  Locked groups survive re-clustering (threshold/model changes); any group
 *  the user edits locks itself automatically. */
interface Group {
  key: string;
  name: string;
  eventIds: string[];
  locked: boolean;
}

export function ReidLabelPage() {
  const qc = useQueryClient();
  const [picked, setPicked] = useState('');
  const [view, setView] = useState<'groups' | 'crops'>('groups');
  const [threshold, setThreshold] = useState<number>(DEFAULT_THRESHOLD);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [statusFilter, setStatusFilter] = useState<'all' | ReidRecord['status']>('all');
  const [groups, setGroups] = useState<Group[]>([]);
  const [dirty, setDirty] = useState(false);
  const [dragOver, setDragOver] = useState<string | null>(null);
  const newGroupSeq = useRef(0);

  const videosQuery = useQuery({
    queryKey: ['reid-videos'],
    queryFn: () => apiFetch<ReidVideo[]>(API.reid.videos),
  });
  const extracted = (videosQuery.data ?? []).filter((v) => v.has_reid);

  const resultsQuery = useQuery({
    queryKey: ['reid-results', picked],
    queryFn: () => apiFetch<{ meta: Record<string, unknown>; records: ReidRecord[] }>(API.reid.results(picked)),
    enabled: Boolean(picked),
  });
  const records = useMemo(() => resultsQuery.data?.records ?? [], [resultsQuery.data]);
  const recordById = useMemo(() => new Map(records.map((r) => [r.id, r])), [records]);

  const clustersQuery = useQuery({
    queryKey: ['reid-clusters', picked, threshold],
    queryFn: () => apiFetch<{ clusters: ReidCluster[] }>(API.reid.clusters(picked, threshold)),
    enabled: Boolean(picked),
  });
  const clusters = useMemo(() => clustersQuery.data?.clusters ?? [], [clustersQuery.data]);

  const playersQuery = useQuery({
    queryKey: ['reid-players', picked],
    queryFn: () => apiFetch<ReidPlayers>(API.reid.players(picked)),
    enabled: Boolean(picked),
  });
  const assignments = useMemo(() => playersQuery.data?.assignments ?? {}, [playersQuery.data]);
  const matches = playersQuery.data?.matches ?? {};

  // (Re)build the board whenever the clustering or saved players change:
  // locked rows carry over untouched, saved players fill in what locked rows
  // don't already hold, fresh clusters cover the rest. Unlocked rows are
  // disposable by design — edits lock their row automatically.
  useEffect(() => {
    if (!picked) return;
    setGroups((prev) => {
      const out: Group[] = prev.filter((g) => g.locked).map((g) => ({ ...g, eventIds: [...g.eventIds] }));
      const covered = new Set(out.flatMap((g) => g.eventIds));

      const byPlayer = new Map<string, string[]>();
      for (const [id, name] of Object.entries(assignments)) {
        if (!covered.has(id) && recordById.has(id)) byPlayer.set(name, [...(byPlayer.get(name) ?? []), id]);
      }
      for (const [name, ids] of [...byPlayer.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
        const lockedSame = out.find((g) => g.name.trim() === name);
        if (lockedSame) {
          lockedSame.eventIds.push(...ids);
        } else {
          out.push({ key: `p:${name}`, name, eventIds: ids, locked: false });
        }
        ids.forEach((id) => covered.add(id));
      }

      for (const c of clusters) {
        const rest = c.event_ids.filter((id) => !covered.has(id));
        if (rest.length) out.push({ key: `c:${threshold}:${c.id}`, name: '', eventIds: rest, locked: false });
      }
      return out;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked, assignments, clusters, recordById]);

  const moveEvent = (eventId: string, fromKey: string, toKey: string) => {
    if (fromKey === toKey) return;
    setGroups((prev) =>
      prev
        .map((g) => {
          if (g.key === fromKey) return { ...g, eventIds: g.eventIds.filter((i) => i !== eventId) };
          // Receiving an edit locks the row so re-clustering can't undo it.
          if (g.key === toKey) return { ...g, eventIds: [...g.eventIds, eventId], locked: true };
          return g;
        })
        // An emptied auto-cluster group is noise; an emptied named player stays.
        .filter((g) => g.eventIds.length > 0 || g.name.trim()),
    );
    setDirty(true);
  };

  /** Merge every event of one row into another; the target's name wins,
   *  falling back to the source's if the target is unnamed. */
  const mergeGroups = (fromKey: string, toKey: string) => {
    if (fromKey === toKey) return;
    setGroups((prev) => {
      const from = prev.find((g) => g.key === fromKey);
      if (!from) return prev;
      return prev
        .filter((g) => g.key !== fromKey)
        .map((g) =>
          g.key === toKey
            ? { ...g, name: g.name.trim() || from.name, eventIds: [...g.eventIds, ...from.eventIds], locked: true }
            : g,
        );
    });
    setDirty(true);
  };

  const toggleLock = (key: string) =>
    setGroups((prev) => prev.map((g) => (g.key === key ? { ...g, locked: !g.locked } : g)));

  const onDropTo = (toKey: string) => (e: DragEvent) => {
    e.preventDefault();
    setDragOver(null);
    const parts = e.dataTransfer.getData('text/plain').split('\n');
    if (parts[0] === 'group') {
      const fromKey = parts[1];
      if (fromKey && toKey !== '__new__') mergeGroups(fromKey, toKey);
      return;
    }
    const [kind, eventId, fromKey] = parts;
    if (kind !== 'event' || !eventId || !fromKey) return;
    if (toKey === '__new__') {
      const key = `n:${newGroupSeq.current++}`;
      setGroups((prev) => [...prev, { key, name: '', eventIds: [], locked: true }]);
      moveEvent(eventId, fromKey, key);
    } else {
      moveEvent(eventId, fromKey, toKey);
    }
  };

  const save = async () => {
    // Locked-but-unnamed rows are curated work — persist them under a
    // placeholder identity (P1, P2, …) the user can rename any time.
    // Untouched auto-clusters stay ephemeral by design.
    const used = new Set(groups.map((g) => g.name.trim()).filter(Boolean));
    let seq = 1;
    const nextPlaceholder = () => {
      while (used.has(`P${seq}`)) seq += 1;
      const name = `P${seq}`;
      used.add(name);
      return name;
    };
    const named = groups.map((g) => (g.locked && !g.name.trim() ? { ...g, name: nextPlaceholder() } : g));

    const next: Record<string, string> = {};
    for (const g of named) {
      const name = g.name.trim();
      if (!name) continue;
      for (const id of g.eventIds) next[id] = name;
    }
    try {
      await apiFetch(API.reid.players(picked), { method: 'PUT', body: { assignments: next } });
      setGroups(named); // show the placeholders the save just minted
      setDirty(false);
      await qc.invalidateQueries({ queryKey: ['reid-players', picked] });
      toast.success(`Saved ${new Set(Object.values(next)).size} player(s), ${Object.keys(next).length} events`);
    } catch (e) {
      toast.error(`Save failed: ${errMsg(e)}`);
    }
  };

  const rebuildFromSaved = () => {
    setGroups([]); // dropping every lock lets the effect rebuild from saved state
    setDirty(false);
    void qc.invalidateQueries({ queryKey: ['reid-clusters', picked] });
  };

  const reset = async () => {
    if (dirty) {
      const ok = await confirm({
        title: 'Discard unsaved changes?',
        body: 'Group edits since the last save (including locks) will be lost.',
        confirmText: 'Discard',
        variant: 'danger',
      });
      if (!ok) return;
    }
    rebuildFromSaved();
  };

  const pickVideo = async (name: string) => {
    if (dirty && name !== picked) {
      const ok = await confirm({
        title: 'Discard unsaved changes?',
        body: 'The current group edits have not been saved.',
        confirmText: 'Discard',
        variant: 'danger',
      });
      if (!ok) return;
    }
    setGroups([]);
    setDirty(false);
    setPicked(name);
  };

  const playerCaption = (r: ReidRecord) => {
    const m = matches[r.id];
    if (!m) return null;
    return (
      <span
        className={cn(
          'truncate font-medium',
          m.assigned ? 'text-primary-light' : m.sim >= LOW_SIM ? 'text-text-secondary' : 'text-text-muted line-through',
        )}
        title={m.assigned ? `${m.player} (labeled)` : `${m.player} · sim ${m.sim.toFixed(2)}`}
      >
        {m.player}
      </span>
    );
  };

  const shown = statusFilter === 'all' ? records : records.filter((r) => r.status === statusFilter);
  const namedCount = groups.filter((g) => g.name.trim()).length;

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <>
            {dirty && <Badge tone="warning">unsaved</Badge>}
            <Button size="sm" onClick={reset} disabled={!dirty}>
              Reset
            </Button>
            <Button intent="primary" onClick={save} disabled={!picked}>
              Save
            </Button>
          </>
        }
      />

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <SectionLabel className="mb-0">Identities</SectionLabel>
            <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
              {(['groups', 'crops'] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setView(v)}
                  className={cn(
                    'rounded-md px-3 py-1 text-xs font-medium capitalize transition-colors',
                    view === v ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]',
                  )}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={showSkeleton}
                onChange={(e) => setShowSkeleton(e.target.checked)}
                className="h-3.5 w-3.5 accent-primary"
              />
              Skeleton
            </label>
            {view === 'groups' ? (
              <select value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className={selectCls} title="Cluster threshold for unassigned events">
                {THRESHOLDS.map((t) => (
                  <option key={t} value={t}>
                    threshold {t}
                  </option>
                ))}
              </select>
            ) : (
              <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)} className={selectCls}>
                <option value="all">All statuses</option>
                <option value="ok">ok</option>
                <option value="multi">multi</option>
                <option value="miss">miss</option>
              </select>
            )}
            <div className="w-80">
              <VideoCombobox items={extracted} value={picked} onChange={pickVideo} placeholder={`Search ${extracted.length} extracted videos…`} />
            </div>
          </div>
        </div>

        {!picked ? (
          <EmptyState
            icon={
              <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.5 20.25a8.25 8.25 0 0115 0" />
              </svg>
            }
            title="Pick an extracted video"
            subtitle="Groups: drag crops between players to fix identities. Crops: review every event's match"
          />
        ) : view === 'groups' ? (
          clustersQuery.isPending || resultsQuery.isPending ? (
            <div className="py-8 text-center text-xs text-text-muted">Clustering…</div>
          ) : (
            <div className="space-y-3">
              <p className="text-xs text-text-muted">
                {namedCount} named player(s) · drag a crop onto another group to reassign it, or drag a row's
                number handle onto another row to merge them (target name wins). Edited rows lock themselves
                (🔒) and survive threshold changes; unlocked rows re-cluster freely. Save persists named rows
                — locked rows without a name get a placeholder (P1, P2, …) you can rename later.
              </p>
              <div className="max-h-[70vh] space-y-2 overflow-auto pr-1">
                {groups.map((g, i) => (
                  <div
                    key={g.key}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(g.key);
                    }}
                    onDragLeave={() => setDragOver((k) => (k === g.key ? null : k))}
                    onDrop={onDropTo(g.key)}
                    className={cn(
                      'rounded-xl border bg-surface-50 p-2.5 transition-colors',
                      dragOver === g.key ? 'border-primary/60 bg-primary/5' : 'border-border',
                    )}
                  >
                    <div className="mb-2 flex items-center gap-3">
                      <span
                        draggable
                        onDragStart={(e) => e.dataTransfer.setData('text/plain', `group\n${g.key}`)}
                        title="Drag onto another row to merge this whole group into it"
                        className={cn(
                          'flex h-6 min-w-8 cursor-grab items-center justify-center rounded-md px-1.5 font-mono text-[11px] font-bold tabular-nums active:cursor-grabbing',
                          g.locked ? 'bg-primary/15 text-primary-light ring-1 ring-primary/25' : 'bg-ink/5 text-text-secondary ring-1 ring-ink/10',
                        )}
                      >
                        {i + 1}
                      </span>
                      <button
                        type="button"
                        onClick={() => toggleLock(g.key)}
                        title={g.locked ? 'Locked — survives threshold/model changes. Click to unlock.' : 'Unlocked — re-clusters on threshold/model change. Click to lock.'}
                        className={cn('flex-shrink-0 transition-colors', g.locked ? 'text-primary-light' : 'text-text-muted hover:text-text-secondary')}
                      >
                        {g.locked ? (
                          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                          </svg>
                        ) : (
                          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 10.5V6.75a4.5 4.5 0 119 0v3.75M3.75 21.75h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H3.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                          </svg>
                        )}
                      </button>
                      <input
                        value={g.name}
                        onChange={(e) => {
                          const name = e.target.value;
                          // Renaming is an edit too — lock so the name sticks.
                          setGroups((prev) => prev.map((x) => (x.key === g.key ? { ...x, name, locked: true } : x)));
                          setDirty(true);
                        }}
                        placeholder="Player name…"
                        className="w-44 rounded-lg border border-border-light bg-surface-100 px-2.5 py-1 text-xs text-text-primary focus:border-primary/50 focus:outline-none"
                      />
                      <span className="font-mono text-[11px] tabular-nums text-text-muted">{g.eventIds.length} events</span>
                      {!g.name.trim() && <Badge tone="neutral">unassigned</Badge>}
                    </div>
                    <div className="flex min-h-[7rem] flex-wrap items-start gap-1.5">
                      {g.eventIds.map((id) => {
                        const r = recordById.get(id);
                        if (!r?.crop) return null;
                        return (
                          <CropImage
                            key={id}
                            src={apiUrl(API.reid.crop(picked, r.crop))}
                            keypoints={r.keypoints}
                            skeleton={showSkeleton}
                            alt={id}
                            draggable
                            onDragStart={(e) => e.dataTransfer.setData('text/plain', `event\n${id}\n${g.key}`)}
                            title={`${r.label} f${r.frame} — drag to move`}
                            className="h-28 w-auto cursor-grab rounded-md border border-border active:cursor-grabbing"
                          />
                        );
                      })}
                    </div>
                  </div>
                ))}
                {/* Drop here to split a crop into a brand-new group */}
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver('__new__');
                  }}
                  onDragLeave={() => setDragOver((k) => (k === '__new__' ? null : k))}
                  onDrop={onDropTo('__new__')}
                  className={cn(
                    'flex items-center justify-center rounded-xl border border-dashed py-6 text-xs transition-colors',
                    dragOver === '__new__' ? 'border-primary/60 bg-primary/5 text-primary-light' : 'border-border text-text-muted',
                  )}
                >
                  Drop a crop here to start a new group
                </div>
              </div>
            </div>
          )
        ) : resultsQuery.isPending ? (
          <div className="py-8 text-center text-xs text-text-muted">Loading…</div>
        ) : (
          <>
            <div className="grid max-h-[70vh] grid-cols-3 gap-2 overflow-auto pr-1 sm:grid-cols-5 lg:grid-cols-8">
              {shown.map((r) => (
                <figure key={r.id} className="overflow-hidden rounded-lg border border-border bg-surface-50">
                  {r.crop ? (
                    <CropImage
                      src={apiUrl(API.reid.crop(picked, r.crop))}
                      keypoints={r.keypoints}
                      skeleton={showSkeleton}
                      alt={`${r.label} f${r.frame}`}
                      className="w-full"
                    />
                  ) : (
                    <div className="flex aspect-[1/2] w-full items-center justify-center text-[10px] text-text-muted">no crop</div>
                  )}
                  <figcaption className="space-y-0.5 px-1.5 py-1 text-[9.5px]">
                    <div className="flex items-center justify-between gap-1">
                      <span className="truncate font-mono text-text-secondary">
                        {r.label} f{r.frame}
                      </span>
                      <span className={cn('h-1.5 w-1.5 flex-shrink-0 rounded-full', STATUS_DOT[r.status])} title={`${r.status} · ${r.candidates} candidate(s)`} />
                    </div>
                    {playerCaption(r)}
                  </figcaption>
                </figure>
              ))}
              {shown.length === 0 && <div className="col-span-full py-8 text-center text-xs text-text-muted">No events match</div>}
            </div>
            {records.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                <Badge tone="success">ok {records.filter((r) => r.status === 'ok').length}</Badge>
                <Badge tone="warning">multi {records.filter((r) => r.status === 'multi').length}</Badge>
                <Badge tone="danger">miss {records.filter((r) => r.status === 'miss').length}</Badge>
                {(playersQuery.data?.players ?? []).map((p) => (
                  <Badge key={p} tone="brand">
                    {p} {Object.values(matches).filter((m) => m.player === p).length}
                  </Badge>
                ))}
              </div>
            )}
          </>
        )}
      </Card>
    </div>
  );
}

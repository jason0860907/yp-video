/** The identities board state machine, extracted from the ReID Label page.
 *
 *  Owns the Group list and everything that mutates it: the rebuild from
 *  clusters + saved assignments, edit operations (move/merge/rename/reorder/
 *  lock), the auto-saving PUT with its in-flight-edit protection, and the
 *  two bulk assists (seeded regroup, tracklet propagation). The view layer
 *  (GroupBoard) only renders and calls these actions.
 */

import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { ReidCluster, ReidRecord } from '@/types/api';
import { errMsg, trackKeyOf, type TrackData } from './shared';

/** One editable identity group: named = a player, unnamed = an auto cluster.
 *  Locked groups survive re-clustering (threshold/model changes); any group
 *  the user edits locks itself automatically. */
export interface Group {
  key: string;
  name: string;
  eventIds: string[];
  locked: boolean;
}

// Auto clusters this small are noise, not players — they pool into one
// shared "unsorted" row instead of each getting its own.
export const MIN_CLUSTER_SIZE = 3;

export interface GroupBoardOptions {
  picked: string;
  embedder: string;
  threshold: number;
  records: ReidRecord[];
  recordById: Map<string, ReidRecord>;
  clusters: ReidCluster[];
  assignments: Record<string, string>;
  /** undefined = still loading, null = no tracking run yet (404). */
  tracks: TrackData | null | undefined;
}

export function useGroupBoard({ picked, embedder, threshold, records, recordById, clusters, assignments, tracks }: GroupBoardOptions) {
  const qc = useQueryClient();
  const [groups, setGroups] = useState<Group[]>([]);
  const [dirty, setDirty] = useState(false);
  // Bumped on every board edit; a save only clears dirty when nothing was
  // edited while its PUT was in flight.
  const editSeq = useRef(0);
  const markDirty = () => {
    editSeq.current += 1;
    setDirty(true);
  };
  const newGroupSeq = useRef(0);

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
          // Saved players are confirmed human work — they come back
          // locked, so a reload looks identical to the pre-save board
          // and re-clustering can never dissolve them.
          out.push({ key: `p:${name}`, name, eventIds: ids, locked: true });
        }
        ids.forEach((id) => covered.add(id));
      }

      const tiny: string[] = [];
      for (const c of clusters) {
        const rest = c.event_ids.filter((id) => !covered.has(id));
        if (!rest.length) continue;
        if (rest.length < MIN_CLUSTER_SIZE) tiny.push(...rest);
        else out.push({ key: `c:${embedder}:${threshold}:${c.id}`, name: '', eventIds: rest, locked: false });
      }
      if (tiny.length) out.push({ key: `pool:${embedder}:${threshold}`, name: '', eventIds: tiny, locked: false });
      return out;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked, assignments, clusters, recordById]);

  /** Move one or many events into ``toKey``, wherever they currently live —
   *  the selection may span multiple source groups. */
  const moveEvents = (eventIds: string[], toKey: string) => {
    const moving = new Set(eventIds);
    if (!moving.size) return;
    setGroups((prev) =>
      prev
        .map((g) => {
          // Receiving an edit locks the row so re-clustering can't undo it.
          if (g.key === toKey) return { ...g, eventIds: [...g.eventIds.filter((i) => !moving.has(i)), ...eventIds], locked: true };
          return { ...g, eventIds: g.eventIds.filter((i) => !moving.has(i)) };
        })
        // An emptied auto-cluster group is noise; an emptied named player stays.
        .filter((g) => g.eventIds.length > 0 || g.name.trim()),
    );
    markDirty();
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
    markDirty();
  };

  /** New empty group inserted right below the group holding anchorId —
   *  keeps the new row next to where the crops came from for comparison. */
  const newGroupBelow = (anchorId: string | undefined) => {
    const key = `n:${newGroupSeq.current++}`;
    setGroups((prev) => {
      const at = anchorId ? prev.findIndex((g) => g.eventIds.includes(anchorId)) : -1;
      const out = [...prev];
      out.splice(at >= 0 ? at + 1 : out.length, 0, { key, name: '', eventIds: [], locked: true });
      return out;
    });
    return key;
  };

  const toggleLock = (key: string) =>
    setGroups((prev) => prev.map((g) => (g.key === key ? { ...g, locked: !g.locked } : g)));

  /** Renaming is an edit too — lock so the name sticks through re-clustering. */
  const renameGroup = (key: string, name: string) => {
    setGroups((prev) => prev.map((g) => (g.key === key ? { ...g, name, locked: true } : g)));
    markDirty();
  };

  /** Move a whole row so it sits before/after targetKey. View-only —
   *  ordering isn't persisted, but locked rows keep it for the session. */
  const reorderGroup = (fromKey: string, targetKey: string, mode: 'before' | 'after') => {
    if (fromKey === targetKey) return;
    setGroups((prev) => {
      const from = prev.find((g) => g.key === fromKey);
      if (!from) return prev;
      const rest = prev.filter((g) => g.key !== fromKey);
      const at = rest.findIndex((g) => g.key === targetKey);
      if (at < 0) return prev;
      rest.splice(mode === 'before' ? at : at + 1, 0, from);
      return rest;
    });
  };

  const savingRef = useRef(false);
  // Save called while a PUT is in flight → run once more when it settles,
  // through the latest closure (the stale one would save stale groups).
  const queuedRef = useRef(false);
  /** Resolves true when THIS call persisted the board (false = failed or
   *  deferred behind an in-flight save) — "Save & Next" gates on it. */
  const save = async (auto = false): Promise<boolean> => {
    if (savingRef.current) {
      queuedRef.current = true;
      return false;
    }
    savingRef.current = true;
    const seq = editSeq.current;
    // Locked-but-unnamed rows are curated work — persist them under a
    // placeholder identity (P1, P2, …) the user can rename any time.
    // Untouched auto-clusters stay ephemeral by design.
    const used = new Set(groups.map((g) => g.name.trim()).filter(Boolean));
    let n = 1;
    const nextPlaceholder = () => {
      while (used.has(`P${n}`)) n += 1;
      const name = `P${n}`;
      used.add(name);
      return name;
    };
    // key → freshly minted placeholder name.
    const minted = new Map<string, string>();
    const named = groups.map((g) => {
      if (!g.locked || g.name.trim()) return g;
      const name = nextPlaceholder();
      minted.set(g.key, name);
      return { ...g, name };
    });

    const next: Record<string, string> = {};
    for (const g of named) {
      const name = g.name.trim();
      if (!name) continue;
      for (const id of g.eventIds) next[id] = name;
    }
    try {
      await apiFetch(API.reid.players(picked, embedder), { method: 'PUT', body: { assignments: next } });
      // Patch the minted placeholders into whatever the board looks like NOW —
      // never replace the array wholesale, edits may have landed mid-PUT and a
      // snapshot would silently revert them.
      if (minted.size) {
        setGroups((cur) => cur.map((g) => (minted.has(g.key) && !g.name.trim() ? { ...g, name: minted.get(g.key)! } : g)));
      }
      // Mid-PUT edits keep the board dirty; the auto-save effect re-fires.
      if (editSeq.current === seq) setDirty(false);
      await qc.invalidateQueries({ queryKey: ['reid-players', picked] });
      if (!auto) toast.success(`Saved ${new Set(Object.values(next)).size} player(s), ${Object.keys(next).length} events`);
      return true;
    } catch (e) {
      toast.error(`Save failed: ${errMsg(e)}`);
      return false;
    } finally {
      savingRef.current = false;
      if (queuedRef.current) {
        queuedRef.current = false;
        void saveRef.current(true);
      }
    }
  };
  const saveRef = useRef(save);
  saveRef.current = save;

  // Auto-save: group edits persist ~1.5 s after the last change; failures
  // leave dirty=true so the next edit (or the Save button) retries.
  useEffect(() => {
    if (!dirty || !picked) return;
    const t = setTimeout(() => void save(true), 1500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dirty, groups, picked]);

  /** Seeded regroup: every locked/named row anchors a player; the backend
   *  assigns all remaining events to the nearest seed centroid (within the
   *  current threshold) and clusters what's left over into fresh pools. */
  const seedRegroup = async () => {
    const anchors = groups.filter((g) => g.locked || g.name.trim());
    const seeds = anchors.filter((g) => g.eventIds.length > 0);
    if (!seeds.length) {
      toast.warning('Lock or name at least one non-empty group to use as a seed');
      return;
    }
    try {
      const res = await apiFetch<{ groups: Record<string, string[]>; leftover_clusters: string[][] }>(
        API.reid.seedCluster(picked),
        { method: 'POST', body: { seeds: Object.fromEntries(seeds.map((g) => [g.key, g.eventIds])), threshold, model: embedder } },
      );
      const assigned = Object.values(res.groups).reduce((s, a) => s + a.length, 0);
      setGroups(() => {
        // Anchors absorb their assignments (locked so nothing re-clusters
        // them away); leftovers become fresh unlocked pools.
        const out = anchors.map((g) => ({ ...g, locked: true, eventIds: [...g.eventIds, ...(res.groups[g.key] ?? [])] }));
        const tiny: string[] = [];
        res.leftover_clusters.forEach((ids, i) => {
          if (ids.length < MIN_CLUSTER_SIZE) tiny.push(...ids);
          else out.push({ key: `seed:${embedder}:${threshold}:${i}`, name: '', eventIds: ids, locked: false });
        });
        if (tiny.length) out.push({ key: `pool:seed:${embedder}:${threshold}`, name: '', eventIds: tiny, locked: false });
        return out;
      });
      markDirty();
      toast.success(`Assigned ${assigned} event(s) to ${seeds.length} seeded group(s) · ${res.leftover_clusters.length} leftover pool(s)`);
    } catch (e) {
      toast.error(`Seed regroup failed: ${errMsg(e)}`);
    }
  };

  /** Track propagate: within a rally, events whose actor boxes lie on the
   *  same ByteTrack tracklet are the same player — every locked/named group
   *  claims the tracklets its members sit on, and unanchored events on a
   *  claimed tracklet follow. Tracklets claimed by two groups are skipped. */
  const trackPropagate = () => {
    if (tracks === null) {
      toast.warning('No tracking for this video yet — run Rally Tracking on the ReID Predict page');
      return;
    }
    if (!tracks) return; // still loading
    const keyOf = (id: string) => trackKeyOf(tracks.links, id);
    const anchors = groups.filter((g) => g.locked || g.name.trim());
    const trackOwner = new Map<string, string>();
    const conflicts = new Set<string>();
    for (const g of anchors) {
      for (const id of g.eventIds) {
        const k = keyOf(id);
        if (!k) continue;
        const owner = trackOwner.get(k);
        if (owner && owner !== g.key) conflicts.add(k);
        else trackOwner.set(k, g.key);
      }
    }
    conflicts.forEach((k) => trackOwner.delete(k));
    const anchored = new Set(anchors.flatMap((g) => g.eventIds));
    const moves = new Map<string, string[]>();
    for (const r of records) {
      if (anchored.has(r.id)) continue;
      const owner = trackOwner.get(keyOf(r.id) ?? '');
      if (owner) moves.set(owner, [...(moves.get(owner) ?? []), r.id]);
    }
    if (!moves.size) {
      toast.warning(
        trackOwner.size
          ? 'No unassigned events share a tracklet with a locked/named group'
          : 'Lock or name a group first — its events anchor the tracklets',
      );
      return;
    }
    let moved = 0;
    for (const [groupKey, ids] of moves) {
      moveEvents(ids, groupKey);
      moved += ids.length;
    }
    toast.success(
      `Propagated ${moved} event(s) along ${trackOwner.size} tracklet(s)` +
        (conflicts.size ? ` · ${conflicts.size} conflicting tracklet(s) skipped` : ''),
    );
  };

  /** Drop every lock and let the rebuild effect restore the saved state. */
  const rebuildFromSaved = () => {
    setGroups([]);
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

  /** Empty the board without touching queries — for video switches. */
  const clearBoard = () => {
    setGroups([]);
    setDirty(false);
  };

  return {
    groups,
    dirty,
    moveEvents,
    mergeGroups,
    newGroupBelow,
    toggleLock,
    renameGroup,
    reorderGroup,
    save,
    seedRegroup,
    trackPropagate,
    reset,
    clearBoard,
  };
}

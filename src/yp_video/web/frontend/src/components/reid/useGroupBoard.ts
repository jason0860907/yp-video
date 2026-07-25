/** The identities board state machine, extracted from the ReID Label page.
 *
 *  Owns the Group list and everything that mutates it: the rebuild from
 *  clusters + saved assignments, edit operations (move/merge/rename/reorder/
 *  lock), the auto-saving PUT with its in-flight-edit protection, and the
 *  two bulk assists (seeded regroup, tracklet propagation). The view layer
 *  (GroupBoard) only renders and calls these actions.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { ReidCluster } from '@/types/api';
import { errMsg } from '@/components/labeling/shared';

/** One editable identity group: named = a player, unnamed = an auto cluster.
 *  Locked groups survive re-clustering (threshold/model changes); any group
 *  the user edits locks itself automatically. */
export interface Group {
  key: string;
  name: string;
  /** UNIT keys, not event ids — a name belongs to the person a tracklet
   *  follows, not to one frame of them (see reid/identity.py). */
  unitKeys: string[];
  locked: boolean;
}

// Auto clusters this small are noise, not players — they pool into one
// shared "unsorted" row instead of each getting its own.
export const MIN_CLUSTER_SIZE = 3;

export interface GroupBoardOptions {
  picked: string;
  embedder: string;
  threshold: number;
  clusters: ReidCluster[];
  /** unit key → the events it covers, for rendering crops. */
  units: Record<string, { event_ids: string[] }>;
  /** unit key → saved player name. */
  unitNames: Record<string, string>;
}

export function useGroupBoard({ picked, embedder, threshold, clusters, units, unitNames }: GroupBoardOptions) {
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

  /** The crops a unit covers, and the reverse — the board holds units, the
   *  tiles it draws are still one per crop. */
  const eventsOf = (unitKey: string): string[] => units[unitKey]?.event_ids ?? [];
  const unitOf = useMemo(() => {
    const map = new Map<string, string>();
    for (const [key, u] of Object.entries(units)) {
      for (const id of u.event_ids) map.set(id, key);
    }
    return map;
  }, [units]);

  // (Re)build the board whenever the clustering or saved players change:
  // locked rows carry over untouched, saved players fill in what those don't
  // already hold, fresh clusters cover the rest. Other unlocked rows are
  // disposable by design — edits lock their row automatically.
  useEffect(() => {
    if (!picked) return;
    setGroups((prev) => {
      const out: Group[] = prev
        .filter((g) => g.locked)
        .map((g) => ({ ...g, unitKeys: [...g.unitKeys] }));
      const covered = new Set(out.flatMap((g) => g.unitKeys));

      const byPlayer = new Map<string, string[]>();
      for (const [key, name] of Object.entries(unitNames)) {
        if (!covered.has(key) && units[key]) byPlayer.set(name, [...(byPlayer.get(name) ?? []), key]);
      }
      for (const [name, ids] of [...byPlayer.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
        const lockedSame = out.find((g) => g.name.trim() === name);
        if (lockedSame) {
          lockedSame.unitKeys.push(...ids);
        } else {
          // Saved players are confirmed human work — they come back
          // locked, so a reload looks identical to the pre-save board
          // and re-clustering can never dissolve them.
          out.push({ key: `p:${name}`, name, unitKeys: ids, locked: true });
        }
        ids.forEach((id) => covered.add(id));
      }

      const tiny: string[] = [];
      for (const c of clusters) {
        const rest = c.unit_keys.filter((key: string) => !covered.has(key));
        if (!rest.length) continue;
        if (rest.length < MIN_CLUSTER_SIZE) tiny.push(...rest);
        else out.push({ key: `c:${embedder}:${threshold}:${c.id}`, name: '', unitKeys: rest, locked: false });
      }
      if (tiny.length) out.push({ key: `pool:${embedder}:${threshold}`, name: '', unitKeys: tiny, locked: false });
      return out;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked, unitNames, clusters, units]);

  /** Move one or many events into ``toKey``, wherever they currently live —
   *  the selection may span multiple source groups. */
  const moveUnits = (unitKeys: string[], toKey: string) => {
    const moving = new Set(unitKeys);
    if (!moving.size) return;
    setGroups((prev) =>
      prev
        .map((g) => {
          // Receiving an edit locks the row so re-clustering can't undo it.
          if (g.key === toKey) return { ...g, unitKeys: [...g.unitKeys.filter((i) => !moving.has(i)), ...unitKeys], locked: true };
          return { ...g, unitKeys: g.unitKeys.filter((i) => !moving.has(i)) };
        })
        // An emptied auto-cluster group is noise; an emptied named player stays.
        .filter((g) => g.unitKeys.length > 0 || g.name.trim()),
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
            ? { ...g, name: g.name.trim() || from.name, unitKeys: [...g.unitKeys, ...from.unitKeys], locked: true }
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
      const at = anchorId ? prev.findIndex((g) => g.unitKeys.includes(anchorId)) : -1;
      const out = [...prev];
      out.splice(at >= 0 ? at + 1 : out.length, 0, { key, name: '', unitKeys: [], locked: true });
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
   *  deferred behind an in-flight save) — the Done button gates on it. */
  const save = async (auto = false): Promise<boolean> => {
    if (savingRef.current) {
      queuedRef.current = true;
      return false;
    }
    // Everything from here runs inside the try: anything that escaped before
    // it would leave savingRef stuck true and silently wedge every future
    // save (the board would look dirty forever, only a reload clearing it).
    savingRef.current = true;
    try {
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

      // A unit key says where its name belongs: a tracklet name covers every
      // action on the track, an event name covers just that event.
      const nextTracks: Record<string, string> = {};
      const nextAssignments: Record<string, string> = {};
      for (const g of named) {
        const name = g.name.trim();
        if (!name) continue;
        for (const key of g.unitKeys) {
          if (key.startsWith('t:')) nextTracks[key.slice(2)] = name;
          else nextAssignments[key.slice(2)] = name;
        }
      }
      await apiFetch(API.reid.players(picked, embedder), {
        method: 'PUT',
        body: { tracks: nextTracks, assignments: nextAssignments },
      });
      // Patch the minted placeholders into whatever the board looks like NOW —
      // never replace the array wholesale, edits may have landed mid-PUT and a
      // snapshot would silently revert them.
      if (minted.size) {
        setGroups((cur) => cur.map((g) => (minted.has(g.key) && !g.name.trim() ? { ...g, name: minted.get(g.key)! } : g)));
      }
      // Mid-PUT edits keep the board dirty; the auto-save effect re-fires.
      if (editSeq.current === seq) setDirty(false);
      await qc.invalidateQueries({ queryKey: ['reid-players', picked] });
      if (!auto) {
        const named = { ...nextTracks, ...nextAssignments };
        toast.success(
          `Saved ${new Set(Object.values(named)).size} player(s) over ` +
            `${Object.keys(nextTracks).length} tracklet(s) and ` +
            `${Object.keys(nextAssignments).length} event(s)`,
        );
      }
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
    const seeds = anchors.filter((g) => g.unitKeys.length > 0);
    if (!seeds.length) {
      toast.warning('Lock or name at least one non-empty group to use as a seed');
      return;
    }
    try {
      const res = await apiFetch<{ groups: Record<string, string[]>; leftover_clusters: string[][] }>(
        API.reid.seedCluster(picked),
        { method: 'POST', body: { seeds: Object.fromEntries(seeds.map((g) => [g.key, g.unitKeys])), threshold, model: embedder } },
      );
      const assigned = Object.values(res.groups).reduce((s, a) => s + a.length, 0);
      setGroups(() => {
        // Anchors absorb their assignments (locked so nothing re-clusters
        // them away); leftovers become fresh unlocked pools.
        const out = anchors.map((g) => ({ ...g, locked: true, unitKeys: [...g.unitKeys, ...(res.groups[g.key] ?? [])] }));
        const tiny: string[] = [];
        res.leftover_clusters.forEach((ids, i) => {
          if (ids.length < MIN_CLUSTER_SIZE) tiny.push(...ids);
          else out.push({ key: `seed:${embedder}:${threshold}:${i}`, name: '', unitKeys: ids, locked: false });
        });
        if (tiny.length) out.push({ key: `pool:seed:${embedder}:${threshold}`, name: '', unitKeys: tiny, locked: false });
        return out;
      });
      markDirty();
      toast.success(`Assigned ${assigned} event(s) to ${seeds.length} seeded group(s) · ${res.leftover_clusters.length} leftover pool(s)`);
    } catch (e) {
      toast.error(`Seed regroup failed: ${errMsg(e)}`);
    }
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
    moveUnits,
    mergeGroups,
    newGroupBelow,
    toggleLock,
    renameGroup,
    reorderGroup,
    save,
    seedRegroup,
    eventsOf,
    unitOf,
    reset,
    clearBoard,
  };
}

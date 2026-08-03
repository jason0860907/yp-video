/** Client-side union of the four labeling work lists, keyed by cut filename.
 *
 *  Fetches with the EXISTING query keys so panel-level invalidations keep
 *  working. Base is `action-videos` — the backend lists every cut there, the
 *  superset for Action/Association/ReID. Rally annotations are keyed by
 *  `<stem>_annotations.jsonl`; ones whose stem matches no cut (R2-only)
 *  become synthetic rally-only rows so they stay pickable.
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import type { RallyResult, UnionVideo } from '@/lib/labelStatus';
import type { ListQuery } from '@/components/video/VideoCombobox';
import type { ActionVideo, AssociationVideo, ReidVideo } from '@/types/api';

const stem = (name: string) => name.replace(/\.[^.]+$/, '');
const rallyStem = (resultName: string) => resultName.replace(/_annotations\.jsonl$/, '');

export function useUnionVideos({ refetchInterval }: { refetchInterval?: number } = {}) {
  const rallyQuery = useQuery({ queryKey: ['annotate-results'], queryFn: () => apiFetch<RallyResult[]>(API.annotate.results), refetchInterval });
  const actionQuery = useQuery({ queryKey: ['action-videos'], queryFn: () => apiFetch<ActionVideo[]>(API.actionAnnotate.videos), refetchInterval });
  const assocQuery = useQuery({ queryKey: ['association-videos'], queryFn: () => apiFetch<AssociationVideo[]>(API.association.videos), refetchInterval });
  const reidQuery = useQuery({ queryKey: ['reid-videos'], queryFn: () => apiFetch<ReidVideo[]>(API.reid.videos), refetchInterval });

  const videos = useMemo<UnionVideo[]>(() => {
    const rows: UnionVideo[] = (actionQuery.data ?? []).map((v) => ({ name: v.name, kind: v.kind, action: v }));
    const byStem = new Map(rows.map((row) => [stem(row.name), row]));
    for (const v of assocQuery.data ?? []) {
      const row = byStem.get(stem(v.name));
      if (row) row.assoc = v;
    }
    for (const v of reidQuery.data ?? []) {
      const row = byStem.get(stem(v.name));
      if (row) row.reid = v;
    }
    for (const r of rallyQuery.data ?? []) {
      const s = rallyStem(r.name);
      const row = byStem.get(s);
      if (row) row.rally = r;
      else rows.push({ name: `${s}.mp4`, kind: r.kind === 'sideline' ? 'sideline' : 'broadcast', rallyOnly: true, rally: r });
    }
    return rows;
  }, [rallyQuery.data, actionQuery.data, assocQuery.data, reidQuery.data]);

  // Rows come from action + rally only (assoc/reid enrich existing rows), so
  // pending tracks just those two — partial data should show, not wait for
  // the slowest list. Errors surface from all four: a failed enrichment
  // silently marks rows "not ready" otherwise.
  const all = [actionQuery, rallyQuery, assocQuery, reidQuery];
  const failed = all.filter((q) => q.isError);
  const query: ListQuery = {
    isPending: actionQuery.isPending || rallyQuery.isPending,
    isError: failed.length > 0,
    error: failed[0]?.error ?? null,
    refetch: () => Promise.all(failed.map((q) => q.refetch())),
  };
  // All four lists have answered at least once — until then, per-mode counts
  // over `videos` would present half-loaded data as real zeros.
  const settled = all.every((q) => !q.isPending);
  return { videos, query, settled };
}

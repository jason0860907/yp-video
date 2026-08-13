import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch } from '@/lib/api';
import type { Rally, SidebarAction, TrackData } from '@/components/labeling/shared';
import type { ReidRecord } from '@/types/api';

interface VideoMeta {
  fps?: number;
  frame_size?: [number, number];
  rallies?: Rally[];
  /** Full action event list (score / non-visible included) served by the
   *  extraction endpoint from the active annotation file. */
  action_events?: Array<Record<string, unknown>>;
}

/** The per-video data layer both labeling pages (ReID, Association) share:
 *  extraction records, tracklets (null until tracked — the 404 is an answer,
 *  not an error) and the full action annotation flattened for the sidebar.
 *  The query keys are shared across the pages too, so switching between them
 *  re-downloads nothing. */
export function useVideoLabelingData(picked: string) {
  const resultsQuery = useQuery({
    queryKey: ['extraction-records', picked],
    queryFn: () =>
      apiFetch<{ meta: Record<string, unknown>; records: ReidRecord[] }>(API.extraction.records(picked)),
    enabled: Boolean(picked),
  });
  const records = useMemo(() => resultsQuery.data?.records ?? [], [resultsQuery.data]);
  const meta = (resultsQuery.data?.meta ?? {}) as VideoMeta;

  const tracksQuery = useQuery({
    queryKey: ['tracklets', picked],
    queryFn: async (): Promise<TrackData | null> => {
      try {
        return await apiFetch<TrackData>(API.tracklets.get(picked));
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
    enabled: Boolean(picked),
    staleTime: 60_000,
  });

  // The full action event list rides on the extraction records response
  // (meta.action_events) — same active annotation file the join uses, so
  // the sidebar and the board can never disagree about the event set.
  const actionEvents = useMemo<SidebarAction[]>(
    () =>
      (meta.action_events ?? []).flatMap((x) => {
        if (x.frame == null) return [];
        const frame = Math.max(0, Math.round(Number(x.frame) || 0));
        return [
          {
            // Same id fallback as the extraction pipeline, so matches line up.
            id: typeof x.id === 'string' && x.id ? x.id : `f${frame}`,
            frame,
            time: typeof x.time === 'number' ? x.time : null,
            label: typeof x.label === 'string' ? x.label : undefined,
            visible: x.visible !== false,
          },
        ];
      }),
    [meta.action_events],
  );

  return { resultsQuery, records, meta, tracksQuery, actionEvents };
}

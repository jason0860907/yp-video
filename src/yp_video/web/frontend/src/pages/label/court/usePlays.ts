/** The play reconstructed on a calibrated court, at the playhead: every
 *  touch placed in court metres (computed server-side, refetched when the
 *  fit moves), the touch whose flight is in the air now, and the flights of
 *  the rally under the playhead. Shared by calibration — the video overlay
 *  checks the marks — and the Analysis page. */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import type { TimelineEvent } from '@/components/editor/ActionTimeline';
import { rallyArcs, type CourtPositions, type CourtState } from './geometry';

export type Position = CourtPositions['events'][number];

export function usePlays(
  video: string,
  state: CourtState | undefined,
  time: number,
  fps: number | undefined,
) {
  const fit = state?.fit ?? null;
  const query = useQuery({
    queryKey: ['court-positions', video, fit?.image_to_court],
    queryFn: () => apiFetch<CourtPositions>(API.court.positions(video)),
    enabled: Boolean(fit),
    retry: false,
  });
  const positions = useMemo(() => query.data?.events ?? [], [query.data]);

  const timelineEvents = useMemo<TimelineEvent[]>(
    () =>
      positions.map((p) => ({
        id: p.id,
        frame: p.frame,
        label: p.label,
        rally_id: p.rally_id,
        visible: p.ball_image != null,
      })),
    [positions],
  );

  const maxFlight = state?.max_flight_s ?? 0;
  // The latest touch not after the playhead, while its flight could still
  // be in the air.
  const current = useMemo(() => {
    const now = time + 0.5 / (fps || 30);
    let hit: Position | null = null;
    for (const p of positions) if (p.time <= now) hit = p;
    return hit && time - hit.time <= maxFlight ? hit : null;
  }, [positions, time, fps, maxFlight]);

  const rallyFlights = useMemo(() => rallyArcs(query.data?.arcs ?? [], time), [query.data, time]);

  return { query, positions, timelineEvents, current, rallyFlights };
}

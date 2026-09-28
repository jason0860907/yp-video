/** Every actor position the server computed, with a download of the same
 *  JSON. Clicking a row seeks the video there. */

import { API, apiUrl, errMsg } from '@/lib/api';
import { actionColor } from '@/lib/actionColors';
import { cn } from '@/lib/cn';
import { SectionLabel } from '@/components/ui/SectionLabel';
import type { CourtPositions } from './geometry';

export function PositionsTable({
  video,
  positions,
  error,
  currentId,
  onSeek,
}: {
  video: string;
  positions: CourtPositions['events'];
  error: unknown;
  currentId: string | null;
  onSeek: (frame: number) => void;
}) {
  return (
    <>
      <div className="mb-2.5 flex items-center justify-between">
        <SectionLabel className="mb-0">Actor positions · {positions.length}</SectionLabel>
        {positions.length > 0 && (
          <a
            href={apiUrl(API.court.positions(video))}
            download={`${video.replace(/\.[^.]+$/, '')}_court_positions.json`}
            className="text-[11px] text-primary hover:underline"
          >
            Download JSON
          </a>
        )}
      </div>
      {error ? (
        <p className="text-[11px] text-amber-400">{errMsg(error)}</p>
      ) : (
        <div className="max-h-72 overflow-y-auto">
          <table className="w-full text-[11px] tabular-nums">
            <thead className="sticky top-0 bg-surface-100 text-text-muted">
              <tr>
                <th className="py-1 text-left font-normal">Time</th>
                <th className="text-left font-normal">Action</th>
                <th className="text-right font-normal">x (m)</th>
                <th className="text-right font-normal">y (m)</th>
                <th className="text-right font-normal" title="Ball height at the touch">
                  z (m)
                </th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr
                  key={p.id}
                  onClick={() => onSeek(p.frame)}
                  className={cn(
                    'cursor-pointer hover:bg-white/5',
                    p.id === currentId && 'bg-white/10',
                    !p.in_court && 'text-text-muted',
                  )}
                >
                  <td className="py-0.5 font-mono">{p.time.toFixed(1)}</td>
                  <td style={{ color: actionColor(p.label) }}>{p.label}</td>
                  <td className="text-right font-mono">{p.court_xy[0].toFixed(2)}</td>
                  <td className="text-right font-mono">{p.court_xy[1].toFixed(2)}</td>
                  <td className="text-right font-mono">
                    {p.ball_3d ? p.ball_3d[2].toFixed(2) : '–'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

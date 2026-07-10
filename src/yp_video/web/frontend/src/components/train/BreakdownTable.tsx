import { cn } from '@/lib/cn';
import type { ActionMapBreakdown, MapBreakdown } from '@/types/api';

// Action breakdowns carry a spatial component (frame-tolerance mAP + pixel
// tolerances); rally segment breakdowns are temporal-only (AP @ tIoU).
const hasSpatial = (bd: MapBreakdown): bd is ActionMapBreakdown => 'spatial' in bd;

export function BreakdownTable({ title, bd, eventNoun = 'events' }: { title: string; bd: MapBreakdown; eventNoun?: string }) {
  const pct = (v: number | undefined) => (Number.isFinite(v) ? ((v as number) * 100).toFixed(1) : '—');
  const numCell = 'py-0.5 pl-5 text-right';
  const spatial = hasSpatial(bd) ? bd.spatial : null;
  return (
    <div className="mt-3">
      <div className="text-xs font-semibold text-text-primary">{title}</div>
      <div className="mt-1.5 grid grid-cols-1 items-start gap-3 xl:grid-cols-[auto_minmax(0,1fr)]">
        {/* AP per class per temporal tolerance */}
        <div className="rounded-lg border border-border bg-surface-100 px-3 py-2.5">
          <div className="text-[9px] uppercase tracking-wider text-text-muted">{spatial ? 'By class (mAP @ tol)' : 'By class (AP @ tIoU)'}</div>
          <table className="mt-1 font-mono text-[10px] tabular-nums">
            <thead>
              <tr className="text-text-muted">
                <th className="py-0.5 text-left font-normal">Class</th>
                {bd.temporal.tolerances.map((t) => (
                  <th key={t} className={cn(numCell, 'font-normal')}>
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(bd.temporal.classes).map(([cls, aps]) => (
                <tr key={cls}>
                  <td className="py-0.5 pr-2 text-left text-text-secondary">{cls}</td>
                  {aps.map((v, i) => (
                    <td key={i} className={cn(numCell, 'text-text-secondary')}>
                      {pct(v)}
                    </td>
                  ))}
                </tr>
              ))}
              <tr className="border-t border-border text-text-primary">
                <td className="py-0.5 pr-2 text-left">overall</td>
                {bd.temporal.overall.map((v, i) => (
                  <td key={i} className={numCell}>
                    {pct(v)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
          {spatial && (
            <div className="mt-1.5 border-t border-border pt-1.5 text-[10px] text-text-muted">
              spatial
              {spatial.pixel_tolerances.map((px, i) => ` ${px}px ${pct(spatial.overall_by_px[i])}`).join('')}
              {' · overall '}
              <span className="text-text-secondary">{pct(spatial.overall)}</span>
            </div>
          )}
        </div>
        {/* By video */}
        {bd.per_video && bd.per_video.length > 0 && (
          <div className="min-w-0 rounded-lg border border-border bg-surface-100 px-3 py-2.5">
            <div className="text-[9px] uppercase tracking-wider text-text-muted">By video</div>
            <table className="mt-1 w-full font-mono text-[10px] tabular-nums">
              <thead>
                <tr className="text-text-muted">
                  <th className="py-0.5 text-left font-normal">Video</th>
                  <th className="w-12 py-0.5 text-right font-normal">mAP</th>
                  {spatial && <th className="w-12 py-0.5 text-right font-normal">temp</th>}
                  {spatial && <th className="w-12 py-0.5 text-right font-normal">spat</th>}
                  <th className="w-14 py-0.5 text-right font-normal">{eventNoun}</th>
                </tr>
              </thead>
              <tbody>
                {[...bd.per_video].sort((a, b) => b.harmonic - a.harmonic).map((v) => (
                  <tr key={v.video}>
                    <td className="max-w-0 truncate py-0.5 pr-3 text-left text-text-secondary" title={v.video}>{v.video}</td>
                    <td className="py-0.5 text-right text-text-primary">{pct(v.harmonic)}</td>
                    {spatial && <td className="py-0.5 text-right text-text-secondary">{pct(v.temporal)}</td>}
                    {spatial && <td className="py-0.5 text-right text-text-secondary">{pct(v.spatial)}</td>}
                    <td className="py-0.5 text-right text-text-secondary">{v.events}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

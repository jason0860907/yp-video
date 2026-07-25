/** Where a video sits in the stage chain, as four chips.
 *
 *  The stages depend on each other but nothing said so: you could start a job
 *  the server rejected two clicks later, and a missing rally source surfaced
 *  as a tracking failure a whole stage away from its cause. These chips put
 *  the answer before the click.
 *
 *  Only the FIRST unmet stage is marked blocked — later gaps are consequences,
 *  and telling someone to run tracking when they have no rallies yet is a
 *  wrong answer, not just an unhelpful one.
 */

import { cn } from '@/lib/cn';
import type { PipelineState } from '@/types/api';

export const STAGE_HINT: Record<NonNullable<PipelineState['blocked_on']>, string> = {
  rallies: 'Label rallies (Rally Label) or run Rally SPOT Predict',
  action: 'Predict or label actions (Action Label)',
  tracks: 'Run Rally Tracking',
  records: 'Run ReID extraction',
};

const STAGES = [
  { key: 'rallies', label: 'Rally', met: (p: PipelineState) => p.rally_sources.length > 0 },
  { key: 'action', label: 'Action', met: (p: PipelineState) => p.has_action },
  { key: 'tracks', label: 'Track', met: (p: PipelineState) => p.has_tracks },
  { key: 'records', label: 'Extract', met: (p: PipelineState) => p.has_records },
] as const;

export function PipelineChips({ pipeline, className }: { pipeline: PipelineState; className?: string }) {
  return (
    <span className={cn('inline-flex flex-wrap items-center gap-1', className)}>
      {STAGES.map(({ key, label, met }) => {
        const done = met(pipeline);
        const blocked = pipeline.blocked_on === key;
        return (
          <span
            key={key}
            title={
              done
                ? `${label}: done`
                : blocked
                  ? `${label} is the next thing missing — ${STAGE_HINT[key]}`
                  : `${label}: waiting on an earlier stage`
            }
            className={cn(
              'rounded px-1.5 py-px font-mono text-[9.5px] uppercase tracking-wide ring-1',
              done && 'bg-primary/12 text-primary-light ring-primary/25',
              blocked && 'bg-amber-500/15 text-amber-400 ring-amber-500/30',
              !done && !blocked && 'bg-ink/5 text-text-muted ring-border',
            )}
          >
            {label}
          </span>
        );
      })}
      {pipeline.tracks_stale && (
        <span
          title="The rallies moved since these tracklets were cut — track ids no longer mean what they meant. Re-run tracking."
          className="rounded bg-red-500/15 px-1.5 py-px font-mono text-[9.5px] uppercase tracking-wide text-red-400 ring-1 ring-red-500/30"
        >
          stale
        </span>
      )}
    </span>
  );
}

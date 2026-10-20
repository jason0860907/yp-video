/** One line of what the App shows around the feeds — angle, date,
 *  court-side score per set, loss reasons, roster — then the worker's jobs
 *  and the projection's warnings. */

import { formatDateTime } from '@/lib/format';
import { JobRow } from './JobStatus';
import type { LibraryMatch, Preview } from './types';
import { lossReasonLabel, playerLabel, sideLabel } from './types';

export function MatchSummary({ match, preview }: { match: LibraryMatch; preview: Preview }) {
  const losses = Object.entries(preview.loss_reasons);
  const parts = [match.angle, formatDateTime(match.recorded_at ?? match.updated_at)];
  return (
    <div className="mt-3 space-y-1.5 border-t border-border pt-3 text-xs text-text-muted">
      <p
        className="flex flex-wrap items-center gap-x-3 gap-y-1"
        title="Not visible here: state kept only on the user's device — watched marks, manual order inside favorites, per-match framing."
      >
        <span>{parts.join(' · ')}</span>
        {preview.court_totals.map((t) => (
          <span
            key={t.set}
            className="text-text-secondary"
            title="Points by the court side that won the rally — not team scores once sides switch"
          >
            Set {t.set}{' '}
            {Object.entries(t.points)
              .map(([side, n]) => `${sideLabel(side)} ${n}`)
              .join(' : ')}
            {t.unknown > 0 && ` (${t.unknown} unknown)`}
          </span>
        ))}
        <span>
          {losses.length === 0
            ? 'No loss reasons'
            : losses.map(([code, n]) => `${lossReasonLabel(code)} ${n}`).join(', ')}
        </span>
        <span title={preview.roster.map(playerLabel).join(', ')}>
          {preview.roster.length === 0 ? 'No roster' : `${preview.roster.length} players`}
        </span>
      </p>
      <JobRow match={match} />
      {preview.warnings.map((w) => (
        <p key={w} className="rounded-md bg-amber-500/10 px-2 py-1.5 text-amber-400">
          {w}
        </p>
      ))}
    </div>
  );
}

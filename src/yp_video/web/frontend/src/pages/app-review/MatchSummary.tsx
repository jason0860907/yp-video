/** The header's match line: angle and date, the worker's jobs, and the
 *  projection's warnings. Scores, loss reasons and players live where the
 *  App shows them — the Score feed and 分析. */

import { formatDateTime } from '@/lib/format';
import { JobRow } from './JobStatus';
import type { LibraryMatch, Preview } from './types';

export function MatchSummary({ match, preview }: { match: LibraryMatch; preview: Preview }) {
  return (
    <div className="mt-3 space-y-1.5 border-t border-border pt-3 text-xs text-text-muted">
      <p
        className="flex flex-wrap items-center gap-x-4 gap-y-1"
        title="Not visible here: state kept only on the user's device — watched marks, manual order inside favorites."
      >
        <span>{[match.angle, formatDateTime(match.recorded_at ?? match.updated_at)].join(' · ')}</span>
        <JobRow match={match} />
      </p>
      {preview.warnings.map((w) => (
        <p key={w} className="rounded-md bg-amber-500/10 px-2 py-1.5 text-amber-400">
          {w}
        </p>
      ))}
    </div>
  );
}

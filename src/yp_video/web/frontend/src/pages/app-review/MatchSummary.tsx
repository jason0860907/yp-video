/** The header's match line: angle and date, then the worker's jobs. Scores,
 *  loss reasons and players live where the App shows them — the Score feed
 *  and 分析. */

import { formatDateTime } from '@/lib/format';
import { JobRow } from './JobStatus';
import type { LibraryMatch } from './types';

export function MatchSummary({ match }: { match: LibraryMatch }) {
  return (
    <p
      className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-text-muted"
      title="Not visible here: state kept only on the user's device — watched marks, manual order inside favorites."
    >
      <span>{[match.angle, formatDateTime(match.recorded_at ?? match.updated_at)].join(' · ')}</span>
      <JobRow match={match} />
    </p>
  );
}

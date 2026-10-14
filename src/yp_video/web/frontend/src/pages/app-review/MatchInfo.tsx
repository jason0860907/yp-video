/** What the App shows around the feeds: roster, court-side score per set,
 *  loss reasons — and what this view cannot show. */

import { SectionLabel } from '@/components/ui/SectionLabel';
import { formatDateTime } from '@/lib/format';
import type { LibraryMatch, Preview } from './types';
import { lossReasonLabel, playerLabel, sideLabel } from './types';

export function MatchInfo({ match, preview }: { match: LibraryMatch; preview: Preview }) {
  return (
    <div className="space-y-5 text-sm">
      <div>
        <SectionLabel>Match</SectionLabel>
        <p className="font-medium">{match.title}</p>
        <p className="text-xs text-text-muted">
          {match.angle} · {formatDateTime(match.recorded_at ?? match.updated_at)} · analysis{' '}
          {match.analysis?.status ?? 'none'}
        </p>
      </div>
      {preview.warnings.length > 0 && (
        <div className="space-y-1">
          {preview.warnings.map((w) => (
            <p key={w} className="rounded-md bg-amber-500/10 px-2 py-1.5 text-xs text-amber-400">
              {w}
            </p>
          ))}
        </div>
      )}
      <div>
        <SectionLabel>Court-side score (by winner side)</SectionLabel>
        {preview.court_totals.length === 0 && <p className="text-text-muted">No rallies.</p>}
        {preview.court_totals.map((t) => (
          <p key={t.set}>
            Set {t.set}:{' '}
            {Object.entries(t.points)
              .map(([side, n]) => `${sideLabel(side)} ${n}`)
              .join(' : ')}
            {t.unknown > 0 && <span className="text-text-muted"> ({t.unknown} unknown)</span>}
          </p>
        ))}
        <p className="mt-1 text-xs text-text-muted">Court-side totals are not team scores once sides switch.</p>
      </div>
      <div>
        <SectionLabel>Loss reasons</SectionLabel>
        {Object.keys(preview.loss_reasons).length === 0 ? (
          <p className="text-text-muted">None marked.</p>
        ) : (
          Object.entries(preview.loss_reasons).map(([code, n]) => (
            <p key={code}>
              {lossReasonLabel(code)} · {n}
            </p>
          ))
        )}
      </div>
      <div>
        <SectionLabel>Roster</SectionLabel>
        {preview.roster.length === 0 ? (
          <p className="text-text-muted">No roster.</p>
        ) : (
          <p>{preview.roster.map(playerLabel).join(', ')}</p>
        )}
      </div>
      <p className="text-xs text-text-muted">
        State kept only on the user's device is not visible here: watched marks, manual order inside favorites,
        per-match framing preferences.
      </p>
    </div>
  );
}

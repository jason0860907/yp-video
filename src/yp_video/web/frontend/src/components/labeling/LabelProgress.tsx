/** The Done / In-Progress / Prediction matrix — one row per label mode, in
 *  pipeline order. Counts come from GET /label/stats, which tallies the same
 *  server-computed statuses the work lists carry, so no surface can disagree
 *  with another — and the sidebar polls one small payload instead of four
 *  full listings.
 *
 *  Self-contained on purpose: label actions and settling jobs invalidate
 *  ['label-stats'] so it refreshes instantly, and it polls to stay fresh for
 *  changes made by other browsers. Rendered in the sidebar footer and on the
 *  Jobs page; react-query dedupes the two instances into one poll.
 */

import { Fragment } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import type { LabelMode } from '@/lib/labelStatus';
import type { LabelStats } from '@/types/api';

const ROWS: Array<[label: string, mode: LabelMode]> = [
  ['Rally', 'rally'],
  ['Action', 'action'],
  ['Assoc', 'association'],
  ['ReID', 'reid'],
];

export function LabelProgress() {
  const { data } = useQuery({
    queryKey: ['label-stats'],
    queryFn: () => apiFetch<LabelStats>(API.label.stats),
    refetchInterval: 30_000,
  });

  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-x-2.5 gap-y-1 text-[11px] text-text-muted">
      <span />
      {['Done', 'In-Prog', 'Pred'].map((h) => (
        <span key={h} className="text-right text-[9px] uppercase tracking-wide">
          {h}
        </span>
      ))}
      {ROWS.map(([label, mode]) => (
        <Fragment key={mode}>
          <span title={`${label}: Done / In-Progress / Prediction (pre-annotate) — counts videos`}>
            {label}
          </span>
          {(['done', 'in-progress', 'pre-annotate'] as const).map((status) => (
            <span key={status} className="text-right font-mono tabular-nums text-text-secondary">
              {data ? data[mode][status] : '–'}
            </span>
          ))}
        </Fragment>
      ))}
    </div>
  );
}

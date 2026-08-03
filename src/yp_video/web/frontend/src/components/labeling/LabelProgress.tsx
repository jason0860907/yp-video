/** The Done / In-Progress / Prediction matrix — one row per label mode, in
 *  pipeline order, counting videos over the same union list the Label page
 *  filters (lib/labelStatus), so no surface can disagree with another.
 *
 *  Self-contained on purpose: it fetches with the Label page's query keys
 *  (label actions refresh it instantly) and keeps itself fresh for changes
 *  made by other browsers. Rendered in the sidebar footer and on the Jobs
 *  page; react-query dedupes the two instances into one poll.
 */

import { Fragment, useMemo } from 'react';
import { countLabelStatuses, type LabelMode } from '@/lib/labelStatus';
import { useUnionVideos } from '@/lib/useUnionVideos';

const ROWS: Array<[label: string, mode: LabelMode]> = [
  ['Rally', 'rally'],
  ['Action', 'action'],
  ['Assoc', 'association'],
  ['ReID', 'reid'],
];

export function LabelProgress() {
  const { videos, settled } = useUnionVideos({ refetchInterval: 30_000 });
  const counts = useMemo(() => countLabelStatuses(videos), [videos]);

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
          {([counts[mode].done, counts[mode]['in-progress'], counts[mode]['pre-annotate']]).map((n, i) => (
            <span key={i} className="text-right font-mono tabular-nums text-text-secondary">
              {settled ? n : '–'}
            </span>
          ))}
        </Fragment>
      ))}
    </div>
  );
}

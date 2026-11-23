/** A user's whole library at a glance: every match with where the worker is
 *  on it and what the user has changed, newest first. A row opens the match;
 *  an analysed one can be sent to training from its row. */

import { useQuery } from '@tanstack/react-query';
import { api, errMsg } from '@/lib/api';
import { formatDateTime } from '@/lib/format';
import { Badge } from '@/components/ui/Badge';
import { ErrorState } from '@/components/ui/ErrorState';
import { IDENTIFY } from './appTerms';
import { JobBadge } from './JobStatus';
import { SendToTraining } from './SendToTraining';
import type { LibraryMatch, MatchEdits } from './types';

/** Overview edit kinds, in the order Review groups them. */
const EDITS: [string, string][] = [
  ['rally', 'Rallies deleted'],
  ['rally_bounds', 'Trims'],
  ['winner', 'Winners'],
  ['actions', 'Actions'],
  ['scores', 'Points'],
  ['player_override', 'Touch players'],
];
/** Match-wide settings: present or not, not counted. */
const SETTINGS: [string, string][] = [
  ['identification', 'Grouping'],
  ['roster', 'Roster'],
];

export function Overview({
  user,
  matches,
  onMatch,
}: {
  user: string;
  matches: LibraryMatch[];
  onMatch: (id: string) => void;
}) {
  const overview = useQuery({
    queryKey: ['app-review', 'overview', user],
    queryFn: () => api.get('/api/app-review/users/{user}/overview', { params: { path: { user } } }),
  });
  if (overview.error) return <ErrorState message={errMsg(overview.error)} onRetry={() => void overview.refetch()} />;

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[1200px] text-left text-[12.5px]">
        <thead>
          <tr className="border-b border-border text-[11px] uppercase tracking-[0.06em] text-text-muted">
            <th className="pb-2 pr-3 font-normal">Match</th>
            <th className="pb-2 pr-3 font-normal">Analysis</th>
            <th className="pb-2 pr-3 font-normal">{IDENTIFY}</th>
            <th className="pb-2 pr-3 font-normal">Court 3D</th>
            <th className="pb-2 pr-3 font-normal">User edits</th>
            <th className="pb-2 pr-3 font-normal">Last edited</th>
            <th className="pb-2 font-normal">Training</th>
          </tr>
        </thead>
        <tbody>
          {matches.map((m) => (
            <tr
              key={m.id}
              onClick={() => onMatch(m.id)}
              className="cursor-pointer border-b border-border align-top last:border-0 hover:bg-surface-100/60"
            >
              <td className="py-2 pr-3">
                <span className="text-text-primary">{m.title}</span>
                {m.owner_id !== user && <Badge className="ml-1.5">shared</Badge>}
                <div className="text-[11px] text-text-muted">{formatDateTime(m.recorded_at ?? m.updated_at)}</div>
              </td>
              <td className="py-2 pr-3">
                <JobBadge job={m.jobs.analysis} />
              </td>
              <td className="py-2 pr-3">
                <JobBadge job={m.jobs.identify} />
              </td>
              <td className="py-2 pr-3">
                <JobBadge job={m.jobs.court} />
              </td>
              <td className="py-2 pr-3">
                <Edits edits={overview.data?.[m.id]} analysed={!!m.r2_keys.result} loading={overview.isPending} />
              </td>
              <td className="whitespace-nowrap py-2 pr-3 text-text-muted">
                {editedAt(overview.data?.[m.id])}
              </td>
              {/* Its own controls: a click here must not open the match. */}
              <td className="whitespace-nowrap py-2" onClick={(e) => e.stopPropagation()}>
                {m.r2_keys.result ? <SendToTraining user={user} match={m} /> : <span className="text-text-muted">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function editedAt(edits: MatchEdits | undefined): string {
  if (!edits || 'error' in edits || !edits.edited_at) return '—';
  return formatDateTime(Date.parse(edits.edited_at) / 1000);
}

function Edits({ edits, analysed, loading }: { edits?: MatchEdits; analysed: boolean; loading: boolean }) {
  if (!analysed) return <span className="text-text-muted">Not analysed yet</span>;
  if (!edits) return <span className="text-text-muted">{loading ? 'Reading…' : '—'}</span>;
  if ('error' in edits) return <span className="text-red-400" title={edits.error}>Could not read</span>;
  const counted = EDITS.filter(([key]) => edits.edits[key]);
  const settings = SETTINGS.filter(([key]) => edits.edits[key]);
  if (!counted.length && !settings.length) return <span className="text-text-muted">No edits</span>;
  return (
    <span className="flex flex-wrap items-center gap-1.5" title={edits.warnings.join('\n') || undefined}>
      {counted.map(([key, label]) => (
        <Badge key={key} tone="accent">
          {label} {edits.edits[key]}
        </Badge>
      ))}
      {settings.map(([key, label]) => (
        <Badge key={key}>{label}</Badge>
      ))}
      {edits.warnings.length > 0 && <Badge tone="warning">warning</Badge>}
    </span>
  );
}

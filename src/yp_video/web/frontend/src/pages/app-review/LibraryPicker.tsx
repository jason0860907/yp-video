/** The header's two pickers: any App user, then that user's library as the
 *  App syncs it (owned and shared matches), newest first. */

import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, errMsg } from '@/lib/api';
import { formatDateTime } from '@/lib/format';
import { SearchSelect } from '@/components/form/SearchSelect';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import type { AppUser, Library } from './types';

const STATUS_TONE: Record<string, BadgeTone> = {
  ready: 'success',
  processing: 'info',
  failed: 'danger',
};

export function LibraryPicker({
  user,
  match,
  library,
  onUser,
  onMatch,
}: {
  user: string;
  match: string;
  library: Library | undefined;
  onUser: (id: string) => void;
  onMatch: (id: string) => void;
}) {
  const users = useQuery({
    queryKey: ['app-review', 'users'],
    queryFn: () => apiFetch<AppUser[]>(API.appReview.users),
  });
  const matches = (library?.matches ?? [])
    .filter((m) => m.deleted_at === null)
    .sort((a, b) => (b.recorded_at ?? b.updated_at) - (a.recorded_at ?? a.updated_at));

  return (
    <>
      <label className="inline-flex min-w-[16rem] flex-1 items-center gap-2 text-xs text-text-muted">
        User
        <SearchSelect
          className="min-w-0 flex-1"
          placeholder={
            users.isError
              ? `Failed to load users: ${errMsg(users.error)}`
              : users.isPending
                ? 'Loading users…'
                : `Search ${users.data?.length ?? 0} users…`
          }
          value={user}
          onChange={onUser}
          options={(users.data ?? []).map((u) => ({
            value: u.id,
            label: u.display_name ?? u.email ?? u.id,
            detail: `${u.email ?? u.id} · ${u.match_count} matches · ${formatDateTime(u.updated_at)} · ${u.provider}`,
          }))}
        />
      </label>
      <label className="inline-flex min-w-[20rem] flex-[2] items-center gap-2 text-xs text-text-muted">
        Match
        <SearchSelect
          className="min-w-0 flex-1"
          placeholder={!user ? 'Pick a user first' : `Search ${matches.length} matches…`}
          value={match}
          onChange={onMatch}
          options={matches.map((m) => ({
            value: m.id,
            label: m.title,
            detail: (
              <span className="flex flex-wrap items-center gap-1.5">
                <Badge tone={STATUS_TONE[m.status] ?? 'neutral'}>{m.status}</Badge>
                {m.r2_keys.corrections && <Badge tone="accent">corrections</Badge>}
                {m.r2_keys.reid && <Badge tone="info">identification</Badge>}
                {m.owner_id !== user && <Badge>shared</Badge>}
                <span>{formatDateTime(m.recorded_at ?? m.updated_at)}</span>
              </span>
            ),
          }))}
        />
      </label>
    </>
  );
}

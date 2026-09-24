/** Left column: every App user, then the chosen user's library as the App
 *  syncs it (owned and shared matches), newest first. */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import type { AppUser, Library } from './types';
import { fmtDate } from './types';

const STATUS_TONE: Record<string, BadgeTone> = {
  ready: 'success',
  processing: 'info',
  failed: 'danger',
};

const rowCls = (active: boolean) =>
  cn(
    'w-full rounded-lg border px-3 py-2 text-left text-sm transition-colors',
    active
      ? 'border-primary/60 bg-primary/10'
      : 'border-transparent hover:border-border-light hover:bg-ink/5',
  );

export function LibraryBrowser({
  user,
  match,
  onUser,
  onMatch,
}: {
  user: string;
  match: string;
  onUser: (id: string) => void;
  onMatch: (id: string) => void;
}) {
  const [search, setSearch] = useState('');
  const users = useQuery({
    queryKey: ['app-review', 'users'],
    queryFn: () => apiFetch<AppUser[]>(API.appReview.users),
  });
  const library = useQuery({
    queryKey: ['app-review', 'library', user],
    enabled: !!user,
    queryFn: () => apiFetch<Library>(API.appReview.library(user)),
  });
  const needle = search.trim().toLowerCase();
  const error = users.error ?? library.error;

  if (!user) {
    const rows = (users.data ?? []).filter(
      (u) => !needle || `${u.email ?? ''} ${u.id}`.toLowerCase().includes(needle),
    );
    return (
      <div className="space-y-2">
        <input
          className={fieldCls}
          placeholder="搜尋 email 或 user id"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {error && <p className="text-sm text-red-400">{errMsg(error)}</p>}
        {users.isLoading && <p className="text-sm text-text-muted">載入用戶…</p>}
        <div className="space-y-1">
          {rows.map((u) => (
            <button
              key={u.id}
              className={rowCls(false)}
              onClick={() => {
                setSearch('');
                onUser(u.id);
              }}
            >
              <span className="block truncate font-medium">{u.email ?? u.id}</span>
              <span className="block text-xs text-text-muted">
                {u.match_count} 場 · {fmtDate(u.updated_at)} · {u.provider}
              </span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  const email = users.data?.find((u) => u.id === user)?.email ?? user;
  const matches = (library.data?.matches ?? []).filter(
    (m) => m.deleted_at === null && (!needle || m.title.toLowerCase().includes(needle)),
  );
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          intent="ghost"
          onClick={() => {
            setSearch('');
            onUser('');
          }}
        >
          ← 用戶
        </Button>
        <span className="truncate text-sm font-medium" title={user}>
          {email}
        </span>
      </div>
      <input
        className={fieldCls}
        placeholder="搜尋比賽"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      {error && <p className="text-sm text-red-400">{errMsg(error)}</p>}
      {library.isLoading && <p className="text-sm text-text-muted">載入比賽…</p>}
      {library.data && matches.length === 0 && (
        <p className="text-sm text-text-muted">此用戶沒有比賽。</p>
      )}
      <div className="space-y-1">
        {matches.map((m) => (
          <button key={m.id} className={rowCls(m.id === match)} onClick={() => onMatch(m.id)}>
            <span className="block truncate font-medium">{m.title}</span>
            <span className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-text-muted">
              <Badge tone={STATUS_TONE[m.status] ?? 'neutral'}>{m.status}</Badge>
              {m.r2_keys.corrections && <Badge tone="accent">有修正</Badge>}
              {m.r2_keys.reid && <Badge tone="info">球員辨識</Badge>}
              {m.owner_id !== user && <Badge>共用</Badge>}
              <span>{fmtDate(m.recorded_at ?? m.updated_at)}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

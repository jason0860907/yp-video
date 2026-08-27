import { useMemo, useState } from 'react';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { API, apiFetch, errMsg } from '@/lib/api';
import { actionLabel, isJobAction, summaryText } from '@/lib/auditLabels';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { ErrorState } from '@/components/ui/ErrorState';
import { PageHeader } from '@/components/ui/PageHeader';
import { StatTile } from '@/components/ui/StatTile';
import { fieldCls } from '@/components/form/Field';
import type {
  AuditEvent,
  AuditFilters,
  AuditPage as AuditPageData,
  AuditChange,
  AuditSaves,
  Worklog,
} from '@/types/api';

const PAGE_SIZE = 100;

/** Monday 00:00 local time, `weeksAgo` weeks back. Settlement happens after a
 *  week closes, so last week has to be reachable, not just this one. */
function weekStart(weeksAgo: number): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  // getDay(): 0 = Sunday. Shift so Monday is the first day.
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7) - weeksAgo * 7);
  return d;
}

const formatHours = (seconds: number): string => {
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return h ? `${h} 小時 ${String(m).padStart(2, '0')} 分` : `${m} 分`;
};

const formatDay = (d: Date): string =>
  d.toLocaleDateString(undefined, { month: '2-digit', day: '2-digit' });

/** Values as they read in a change line: seconds as mm:ss, everything else
 *  compactly. Rally boundaries are seconds, and "83" tells you far less than
 *  "1:23" when you are checking someone's edit. */
function formatValue(field: string, v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'number' && (field === 'start' || field === 'end' || field === 'time')) {
    const m = Math.floor(v / 60);
    const sec = v % 60;
    return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
  }
  if (typeof v === 'number') return String(Math.round(v * 1000) / 1000);
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) return `[${v.map((x) => formatValue('', x)).join(', ')}]`;
  return JSON.stringify(v);
}

const OP_LABEL: Record<AuditChange['op'], string> = {
  added: '新增',
  removed: '刪除',
  edited: '修改',
  truncated: '其餘',
};

const OP_TONE: Record<AuditChange['op'], 'success' | 'danger' | 'info' | 'neutral'> = {
  added: 'success',
  removed: 'danger',
  edited: 'info',
  truncated: 'neutral',
};

/** One changed item, e.g. 「修改 #3  結束 0:45.2 → 0:47.8」. */
function ChangeLine({ change }: { change: AuditChange }) {
  if (change.op === 'truncated') {
    return (
      <span className="text-[11px] text-text-muted">
        還有 {change.count} 項變更未列出
      </span>
    );
  }
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-1.5 text-[11px]">
      <Badge tone={OP_TONE[change.op]}>{OP_LABEL[change.op]}</Badge>
      <span className="font-mono text-text-secondary">#{change.id}</span>
      {change.fields &&
        Object.entries(change.fields).map(([f, [was, now]]) => (
          <span key={f} className="font-mono tabular-nums text-text-muted">
            {f} {formatValue(f, was)} <span className="text-text-muted/60">→</span>{' '}
            <span className="text-text-primary">{formatValue(f, now)}</span>
          </span>
        ))}
      {change.item && (
        <span className="font-mono tabular-nums text-text-muted">
          {Object.entries(change.item)
            .filter(([f]) => f !== 'id' && f !== 'rally_id')
            .map(([f, v]) => `${f} ${formatValue(f, v)}`)
            .join('  ')}
        </span>
      )}
    </span>
  );
}

/** Every save folded into one row, fetched on demand: when, and what changed.
 *
 *  The gap since the previous save is shown too — every gap here is under the
 *  idle limit, because a longer one would have started a new row. */
function SavesDetail({ event }: { event: AuditEvent }) {
  const saves = useQuery({
    queryKey: ['audit-saves', event.id],
    queryFn: () => apiFetch<AuditSaves>(API.audit.saves(event.id)),
    staleTime: Infinity,
  });

  if (saves.isError) {
    return <span className="text-[11px] text-red-400">{errMsg(saves.error)}</span>;
  }
  if (!saves.data) {
    return <span className="text-[11px] text-text-muted">載入中…</span>;
  }
  if (!saves.data.saves.length) {
    return (
      <span className="text-[11px] text-text-muted">
        這筆紀錄早於變更明細功能，只有時間與統計。
      </span>
    );
  }

  return (
    <div className="flex flex-col gap-1.5 border-l border-border pl-3">
      {saves.data.saves.map((save, i) => {
        const prev = i > 0 ? saves.data.saves[i - 1] : null;
        const gapMs = prev ? new Date(save.at).getTime() - new Date(prev.at).getTime() : 0;
        return (
          <div key={save.at + i} className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="font-mono text-[11px] tabular-nums text-text-muted">
              {new Date(save.at).toLocaleTimeString(undefined, CLOCK)}
              {prev && (
                <span className="ml-1 text-text-muted/50">
                  +{gapMs < 60_000 ? `${Math.round(gapMs / 1000)}s` : `${Math.round(gapMs / 60_000)}m`}
                </span>
              )}
            </span>
            {save.changes.length ? (
              save.changes.map((c, ci) => <ChangeLine key={ci} change={c} />)
            ) : (
              <span className="text-[11px] text-text-muted/60">（無變更明細）</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Labeling time per person for one week.
 *
 *  How the numbers are built, since the card deliberately does not say so on
 *  screen — the reader wants the figure, not a lecture under every table:
 *
 *  - A session is a run of one person's saves — any video, any editor — in
 *    which no two consecutive saves are more than ten minutes apart
 *    (SESSION_IDLE_GAP in web/audit.py). `工時` sums first-to-last save of
 *    each session, so the quiet between two videos counts as work while a
 *    lunch break does not. Sessions are NOT the rows of the trail above:
 *    those fold per video.
 *  - Only labeling actions count (audit.LABELING_ACTIONS). A publish, a
 *    delete or a training start is instantaneous and contributes nothing.
 *  - A session's clock starts at its first SAVE, not its first edit, so each
 *    session loses the ~2 s autosave debounce. A session with a single save
 *    spans zero — which is why `段數` and `存檔次數` sit beside `工時`.
 */
function WorklogCard() {
  const [weeksAgo, setWeeksAgo] = useState(0);
  const since = useMemo(() => weekStart(weeksAgo), [weeksAgo]);
  const until = useMemo(() => weekStart(weeksAgo - 1), [weeksAgo]);

  const log = useQuery({
    queryKey: ['audit-worklog', weeksAgo],
    queryFn: () =>
      apiFetch<Worklog>(
        API.audit.worklog({ since: since.toISOString(), until: until.toISOString() }),
      ),
  });

  return (
    <Card
      label={`工時 · ${formatDay(since)}–${formatDay(new Date(until.getTime() - 1))}`}
      className="mb-4"
      right={
        <div className="flex items-center gap-1.5">
          {[0, 1].map((n) => (
            <Button
              key={n}
              size="sm"
              intent={weeksAgo === n ? 'primary' : 'ghost'}
              onClick={() => setWeeksAgo(n)}
            >
              {n === 0 ? '本週' : '上週'}
            </Button>
          ))}
        </div>
      }
    >
      {log.isError ? (
        <ErrorState message={errMsg(log.error)} onRetry={() => void log.refetch()} />
      ) : !log.data?.people.length ? (
        <p className="py-4 text-center text-xs text-text-muted">
          {log.isLoading ? '載入中…' : '這週還沒有標註紀錄'}
        </p>
      ) : (
        <>
          <table className="w-full text-left text-[12.5px]">
            <thead>
              <tr className="border-b border-border text-[11px] uppercase tracking-[0.06em] text-text-muted">
                <th className="pb-2 pr-3 font-normal">執行者</th>
                <th className="pb-2 pr-3 text-right font-normal">工時</th>
                <th className="pb-2 pr-3 text-right font-normal">段數</th>
                <th className="pb-2 text-right font-normal">存檔次數</th>
              </tr>
            </thead>
            <tbody>
              {log.data.people.map((p) => (
                <tr key={p.actor} className="border-b border-border-light/60 last:border-0">
                  <td className="py-2 pr-3 text-text-secondary">{p.actor}</td>
                  <td className="py-2 pr-3 text-right font-mono tabular-nums text-text-primary">
                    {formatHours(p.seconds)}
                  </td>
                  <td className="py-2 pr-3 text-right font-mono tabular-nums text-text-muted">
                    {p.sessions}
                  </td>
                  <td className="py-2 text-right font-mono tabular-nums text-text-muted">
                    {p.saves}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Card>
  );
}

interface Filters {
  actor: string;
  action: string;
  target: string;
  since: string;
  until: string;
}

const EMPTY: Filters = { actor: '', action: '', target: '', since: '', until: '' };

/** A datetime-local value is wall-clock with no zone; the backend compares in
 *  UTC, so hand it a real instant rather than a naive string. */
const asInstant = (local: string): string | undefined =>
  local ? new Date(local).toISOString() : undefined;

const DATE_TIME: Intl.DateTimeFormatOptions = {
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
};

const CLOCK: Intl.DateTimeFormatOptions = {
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
};

const formatWhen = (iso: string): string => new Date(iso).toLocaleString(undefined, DATE_TIME);

/** The end only needs the clock when it falls on the same day as the start. */
function formatEnd(startIso: string, endIso: string): string {
  const start = new Date(startIso);
  const end = new Date(endIso);
  return start.toDateString() === end.toDateString()
    ? end.toLocaleTimeString(undefined, CLOCK)
    : end.toLocaleString(undefined, DATE_TIME);
}

/** How long this session ran: end − start. "—" for a one-shot action, whose
 *  start and end are the same instant. */
function formatSpan(startIso: string, endIso: string): string {
  const ms = new Date(endIso).getTime() - new Date(startIso).getTime();
  if (ms < 1000) return '—';
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
}

export function AuditPage() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [draft, setDraft] = useState<Filters>(EMPTY);
  // Applied separately so typing a target doesn't refetch on every keystroke.
  const [applied, setApplied] = useState<Filters>(EMPTY);

  const filters = useQuery({
    queryKey: ['audit-filters'],
    queryFn: () => apiFetch<AuditFilters>(API.audit.filters),
  });

  const events = useInfiniteQuery({
    queryKey: ['audit-events', applied],
    initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam }) =>
      apiFetch<AuditPageData>(
        API.audit.events({
          actor: applied.actor || undefined,
          action: applied.action || undefined,
          target: applied.target || undefined,
          since: asInstant(applied.since),
          until: asInstant(applied.until),
          before: pageParam,
          limit: PAGE_SIZE,
        }),
      ),
    getNextPageParam: (last) => last.next_before ?? undefined,
  });

  const rows: AuditEvent[] = useMemo(
    () => events.data?.pages.flatMap((p) => p.events) ?? [],
    [events.data],
  );

  // Every tile describes the rows currently loaded under the applied filters,
  // never the whole table — a stat that ignores the filter above it lies.
  const stats = useMemo(() => {
    const actors = new Set(rows.map((r) => r.actor));
    const failures = rows.filter((r) => r.outcome === 'error').length;
    const saves = rows.reduce((n, r) => n + r.repeats, 0);
    return { loaded: rows.length, actors: actors.size, failures, saves };
  }, [rows]);

  const dirty = JSON.stringify(draft) !== JSON.stringify(applied);
  const set = (patch: Partial<Filters>) => setDraft((d) => ({ ...d, ...patch }));

  return (
    <>
      <PageHeader subtitle="每個會改變狀態的操作與背景工作，記錄執行者、對象與結果。編輯器的自動存檔會摺疊成一列，次數顯示為 ×N。" />

      <WorklogCard />

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="事件" value={stats.loaded} tintClass="text-primary-light" />
        <StatTile label="操作次數" value={stats.saves} tintClass="text-primary-light" />
        <StatTile label="執行者" value={stats.actors} tintClass="text-primary-light" />
        <StatTile
          label="失敗"
          value={stats.failures}
          tintClass={stats.failures ? 'text-red-400' : 'text-primary-light'}
        />
      </div>

      <Card label="Filters" className="mb-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] text-text-muted">執行者</span>
            <select
              className={cn(fieldCls, 'cursor-pointer appearance-none')}
              value={draft.actor}
              onChange={(e) => set({ actor: e.target.value })}
            >
              <option value="">全部</option>
              {filters.data?.actors.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] text-text-muted">動作</span>
            <select
              className={cn(fieldCls, 'cursor-pointer appearance-none')}
              value={draft.action}
              onChange={(e) => set({ action: e.target.value })}
            >
              <option value="">全部</option>
              {filters.data?.actions.map((a) => (
                <option key={a} value={a}>
                  {actionLabel(a)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] text-text-muted">對象（影片／類別）</span>
            {/* A datalist, not a select: the list is every video the trail has
                seen, and partial text still matches server-side (ILIKE). */}
            <input
              className={fieldCls}
              list="audit-targets"
              value={draft.target}
              placeholder={filters.data?.targets[0] ?? '影片名稱，可只打一段'}
              onChange={(e) => set({ target: e.target.value })}
            />
            <datalist id="audit-targets">
              {filters.data?.targets.map((t) => (
                <option key={t} value={t} />
              ))}
            </datalist>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] text-text-muted">起</span>
            <input
              type="datetime-local"
              className={fieldCls}
              value={draft.since}
              onChange={(e) => set({ since: e.target.value })}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] text-text-muted">迄</span>
            <input
              type="datetime-local"
              className={fieldCls}
              value={draft.until}
              onChange={(e) => set({ until: e.target.value })}
            />
          </label>
        </div>
        <div className="mt-3 flex items-center gap-2.5">
          <Button size="sm" intent="primary" disabled={!dirty} onClick={() => setApplied(draft)}>
            套用
          </Button>
          <Button
            size="sm"
            intent="ghost"
            onClick={() => {
              setDraft(EMPTY);
              setApplied(EMPTY);
            }}
          >
            清除
          </Button>
        </div>
      </Card>

      <Card label="Events">
        {events.isError ? (
          <ErrorState message={errMsg(events.error)} onRetry={() => void events.refetch()} />
        ) : rows.length === 0 && !events.isLoading ? (
          <EmptyState
            icon={
              <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            }
            title="沒有符合條件的事件"
            subtitle="調整上方篩選條件，或先在其他頁面做一次操作。"
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1040px] text-left text-[12.5px]">
              <thead>
                <tr className="border-b border-border text-[11px] uppercase tracking-[0.06em] text-text-muted">
                  <th className="pb-2 pr-3 font-normal">開始</th>
                  <th className="pb-2 pr-3 font-normal">結束</th>
                  <th className="pb-2 pr-3 font-normal">時長</th>
                  <th className="pb-2 pr-3 font-normal">執行者</th>
                  <th className="pb-2 pr-3 font-normal">動作</th>
                  <th className="pb-2 pr-3 font-normal">對象</th>
                  <th className="pb-2 pr-3 font-normal">摘要</th>
                  <th className="pb-2 font-normal">結果</th>
                </tr>
              </thead>
              <tbody>
                {rows.flatMap((row) => [
                  <tr key={row.id} className="border-b border-border-light/60">
                    <td className="whitespace-nowrap py-2 pr-3 font-mono tabular-nums text-text-muted">
                      {formatWhen(row.first_at)}
                    </td>
                    <td className="whitespace-nowrap py-2 pr-3 font-mono tabular-nums text-text-muted">
                      {formatEnd(row.first_at, row.at)}
                    </td>
                    <td className="whitespace-nowrap py-2 pr-3 font-mono tabular-nums text-text-secondary">
                      {formatSpan(row.first_at, row.at)}
                    </td>
                    <td className="py-2 pr-3 text-text-secondary">{row.actor}</td>
                    <td className="py-2 pr-3">
                      <span
                        className={cn(
                          isJobAction(row.action) ? 'text-text-muted' : 'text-text-primary',
                        )}
                      >
                        {actionLabel(row.action)}
                      </span>
                      {row.repeats > 1 && (
                        <button
                          type="button"
                          title="展開每一次存檔的時間"
                          aria-expanded={expanded === row.id}
                          onClick={() => setExpanded(expanded === row.id ? null : row.id)}
                          className="ml-2 align-middle"
                        >
                          <Badge
                            tone={expanded === row.id ? 'brand' : 'neutral'}
                            className="cursor-pointer hover:bg-ink/10"
                          >
                            ×{row.repeats}
                          </Badge>
                        </button>
                      )}
                    </td>
                    <td className="max-w-[220px] truncate py-2 pr-3 font-mono text-text-secondary" title={row.target ?? ''}>
                      {row.target ?? '—'}
                    </td>
                    <td className="max-w-[260px] truncate py-2 pr-3 font-mono text-[11.5px] text-text-muted">
                      {summaryText(row.summary)}
                    </td>
                    {/* Outcome only. The request latency lives in the row
                        (duration_ms) but showing it beside 時長 invited reading
                        one as the other — they measure different things. */}
                    <td className="whitespace-nowrap py-2">
                      {row.outcome === 'error' ? (
                        <Badge tone="danger">{row.status ?? 'error'}</Badge>
                      ) : (
                        <span className="font-mono text-[11px] text-text-muted">ok</span>
                      )}
                    </td>
                  </tr>,
                  expanded === row.id ? (
                    <tr key={`${row.id}-saves`} className="border-b border-border-light/60">
                      <td colSpan={8} className="px-1 pb-3 pt-0.5">
                        <SavesDetail event={row} />
                      </td>
                    </tr>
                  ) : null,
                ])}
              </tbody>
            </table>
            {events.hasNextPage && (
              <div className="mt-3 flex justify-center">
                <Button
                  size="sm"
                  intent="ghost"
                  disabled={events.isFetchingNextPage}
                  onClick={() => void events.fetchNextPage()}
                >
                  {events.isFetchingNextPage ? '載入中…' : '載入更多'}
                </Button>
              </div>
            )}
          </div>
        )}
      </Card>
    </>
  );
}

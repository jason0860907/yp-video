import { useQuery } from '@tanstack/react-query';
import { NavLink } from 'react-router-dom';
import { API, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Icon } from '@/components/ui/Icon';
import type { ActiveCount, SystemStats, VllmStatus } from '@/types/api';
import { NAV } from './nav';

const VLLM_UI: Record<VllmStatus['status'], { dot: string; label: string }> = {
  running: { dot: 'bg-emerald-400 shadow-[0_0_6px_rgba(52,199,89,0.5)]', label: 'vLLM: running' },
  starting: { dot: 'bg-amber-400 animate-pulse-dot', label: 'vLLM: starting…' },
  stopped: { dot: 'bg-text-muted', label: 'vLLM: stopped' },
  error: { dot: 'bg-red-400 shadow-[0_0_6px_rgba(255,69,58,0.5)]', label: 'vLLM: error' },
};

const STAT_ROWS: Array<[label: string, key: keyof SystemStats]> = [
  ['Videos', 'videos'],
  ['Cuts', 'cuts'],
  ['Rally-Pred', 'pre_annotations'],
  ['Rally Labels', 'annotations'],
  ['Action-Pred', 'action_pre_annotations'],
  ['Action Labels', 'actions'],
  ['VJEPA-B', 'vjepa_b'],
  ['TAD-Pred', 'predictions'],
];

export function Sidebar() {
  const vllm = useQuery({
    queryKey: ['vllm-status'],
    queryFn: () => apiFetch<VllmStatus>(API.system.vllmStatus),
    refetchInterval: 30_000,
  });
  const stats = useQuery({
    queryKey: ['system-stats'],
    queryFn: () => apiFetch<SystemStats>(API.system.stats),
    refetchInterval: 30_000,
  });
  const jobs = useQuery({
    queryKey: ['jobs-active-count'],
    queryFn: () => apiFetch<ActiveCount>(API.jobs.activeCount),
    refetchInterval: 30_000,
  });

  const jobCount = jobs.data?.count ?? 0;
  const vllmUi = VLLM_UI[vllm.data?.status ?? 'stopped'];

  return (
    <aside className="flex h-full w-[212px] flex-shrink-0 flex-col border-r border-border bg-sidebar">
      {/* nav */}
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 pt-3">
        {NAV.map((section) => (
          <div key={section.title}>
            <p className="px-3 pb-1.5 pt-3 text-[10px] font-semibold uppercase tracking-widest text-text-muted">
              {section.title}
            </p>
            {section.items.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-lg border-l-2 px-3 py-2 text-sm transition-colors duration-150',
                    isActive
                      ? 'border-primary bg-gradient-to-r from-primary/20 to-transparent text-text-primary'
                      : 'border-transparent text-text-secondary hover:bg-ink/5 hover:text-text-primary',
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon paths={item.icon} className={isActive ? 'text-primary-light' : undefined} />
                    {item.label}
                    {item.path === '/jobs' && jobCount > 0 && (
                      <span className="ml-auto flex h-5 w-5 items-center justify-center rounded-full bg-accent text-[10px] font-semibold text-surface">
                        {jobCount}
                      </span>
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      {/* footer: vLLM status + stats */}
      <div className="mx-2 mb-3 space-y-2.5 rounded-xl border border-border bg-surface-100 p-3">
        <NavLink to="/jobs" className="flex items-center gap-2 text-xs text-text-muted transition-colors hover:text-text-secondary" title="vLLM status">
          <span className={cn('h-2 w-2 flex-shrink-0 rounded-full ring-2 ring-surface-100', vllmUi.dot)} />
          <span className="font-mono text-[11px]">{vllmUi.label}</span>
        </NavLink>
        <div className="h-px w-full bg-border" />
        <div className="space-y-1 text-[11px] text-text-muted">
          {STAT_ROWS.map(([label, key]) => (
            <div key={key} className="flex items-center justify-between">
              <span>{label}</span>
              <span className="font-mono tabular-nums text-text-secondary">{stats.data?.[key] ?? 0}</span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

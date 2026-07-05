import { cn } from '@/lib/cn';

const MAP: Record<string, string> = {
  running: 'bg-primary/20 text-primary-light ring-1 ring-primary/30',
  completed: 'bg-primary/15 text-primary-light ring-1 ring-primary/25',
  failed: 'bg-red-500/15 text-red-400 ring-1 ring-red-500/20',
  cancelled: 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20',
  pending: 'bg-ink/5 text-text-muted ring-1 ring-ink/10',
  stopped: 'bg-ink/5 text-text-muted ring-1 ring-ink/10',
};

/** Job/server status pill. Lowercase label + color; running pulses. */
export function StatusBadge({ status }: { status: string }) {
  const cls = MAP[status] ?? MAP.pending;
  return (
    <span className={cn('inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium', cls)}>
      <span className={cn('h-1.5 w-1.5 rounded-full bg-current', status === 'running' && 'animate-pulse-dot')} />
      {status}
    </span>
  );
}

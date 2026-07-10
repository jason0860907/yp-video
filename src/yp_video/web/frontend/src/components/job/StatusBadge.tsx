import { cn } from '@/lib/cn';
import { statusTheme } from '@/lib/job';

/** Job/server status pill. Lowercase label + color; running pulses. */
export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium', statusTheme(status).pill)}>
      <span className={cn('h-1.5 w-1.5 rounded-full bg-current', status === 'running' && 'animate-pulse-dot')} />
      {status}
    </span>
  );
}

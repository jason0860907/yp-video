import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface StatTileProps {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  /** Tailwind text-color class for the number (e.g. `text-primary-light`). */
  tintClass?: string;
}

/** KPI tile: small label, big mono number, optional sub. Numbers are mono. */
export function StatTile({ label, value, sub, tintClass = 'text-text-primary' }: StatTileProps) {
  return (
    <div className="card px-[17px] py-[15px]">
      <div className="font-body text-[11px] text-text-muted">{label}</div>
      <div className={cn('mt-1 font-mono text-[26px] font-bold tabular-nums', tintClass)}>{value}</div>
      {sub && <div className="mt-0.5 truncate text-[10.5px] text-text-muted">{sub}</div>}
    </div>
  );
}

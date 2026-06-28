import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

export type BadgeTone = 'neutral' | 'brand' | 'accent' | 'success' | 'warning' | 'danger' | 'info';

const TONES: Record<BadgeTone, string> = {
  neutral: 'text-text-muted bg-ink/5',
  brand: 'text-primary-light bg-primary/20',
  accent: 'text-accent bg-accent/15',
  success: 'text-emerald-400 bg-emerald-500/15',
  warning: 'text-amber-400 bg-amber-500/15',
  danger: 'text-red-400 bg-red-500/15',
  info: 'text-sky-400 bg-sky-500/15',
};

interface BadgeProps {
  children: ReactNode;
  tone?: BadgeTone;
  /** Animated pulse dot prefix — for live/running status. */
  dot?: boolean;
  className?: string;
}

/** Small uppercase status pill. Color carries meaning; no glyphs. */
export function Badge({ children, tone = 'neutral', dot = false, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 align-middle',
        'font-mono text-[9.5px] font-bold uppercase leading-relaxed tracking-[0.02em]',
        TONES[tone],
        className,
      )}
    >
      {dot && <span className="h-1.5 w-1.5 flex-shrink-0 rounded-full bg-current animate-pulse-dot" />}
      {children}
    </span>
  );
}

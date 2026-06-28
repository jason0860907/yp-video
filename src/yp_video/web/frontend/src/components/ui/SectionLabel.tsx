import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

/** Small uppercase section eyebrow above a list/panel — the prototype's
 *  `_sectionLabel`. */
export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('mb-2.5 font-body text-[11px] uppercase tracking-[0.06em] text-text-muted', className)}>
      {children}
    </div>
  );
}

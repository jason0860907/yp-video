import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

/** Native-details disclosure with a SectionLabel-styled summary — the train
 *  pages' "Advanced" parameter fold. */
export function Collapsible({
  label,
  defaultOpen = false,
  className,
  children,
}: {
  label: string;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details open={defaultOpen} className={cn('group', className)}>
      <summary className="flex cursor-pointer select-none list-none items-center gap-1.5 font-body text-[11px] uppercase tracking-[0.06em] text-text-muted hover:text-text-secondary [&::-webkit-details-marker]:hidden">
        <svg
          viewBox="0 0 20 20"
          fill="currentColor"
          className="h-3 w-3 transition-transform group-open:rotate-90"
        >
          <path d="M7 5l6 5-6 5V5z" />
        </svg>
        {label}
      </summary>
      <div className="mt-3">{children}</div>
    </details>
  );
}

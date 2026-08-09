/** The labelled wrapper every schema-bound field renders into: the uppercase
 *  eyebrow label plus, when the contract carries a description, an ⓘ whose
 *  hover/focus popover shows it. CSS-only — no portal, no positioning lib.
 */

import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

function InfoDot({ description }: { description: string }) {
  return (
    <span className="group/info relative inline-flex">
      <span
        tabIndex={0}
        aria-label={description}
        className="cursor-help text-[10px] leading-none text-text-muted/60 hover:text-text-secondary focus:text-text-secondary focus:outline-none"
      >
        ⓘ
      </span>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute left-0 top-full z-40 mt-1.5 w-60 rounded-lg border border-border bg-surface-100 p-2.5 text-[11px] font-normal normal-case leading-relaxed tracking-normal text-text-secondary opacity-0 shadow-xl transition-opacity group-hover/info:visible group-hover/info:opacity-100 group-focus-within/info:visible group-focus-within/info:opacity-100"
      >
        {description}
      </span>
    </span>
  );
}

export function FieldShell({
  label,
  description,
  className,
  children,
}: {
  label: string;
  description?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <label className={cn('block min-w-0 space-y-1', className)}>
      <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest text-text-muted">
        {label}
        {description && <InfoDot description={description} />}
      </span>
      {children}
    </label>
  );
}

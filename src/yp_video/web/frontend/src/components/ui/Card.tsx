import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Optional uppercase eyebrow label in the header row. */
  label?: ReactNode;
  /** Optional right-aligned header slot (actions, badge, …). */
  right?: ReactNode;
}

/** Flat iOS panel — the default building block for every surface. Hairline
 *  border, no shadow (the video console is the only elevated surface). */
export function Card({ label, right, className, children, ...rest }: CardProps) {
  return (
    <div className={cn('card p-[18px]', className)} {...rest}>
      {(label || right) && (
        <div className="mb-3 flex items-center justify-between">
          {label && (
            <span className="font-body text-[11px] uppercase tracking-[0.06em] text-text-muted">
              {label}
            </span>
          )}
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

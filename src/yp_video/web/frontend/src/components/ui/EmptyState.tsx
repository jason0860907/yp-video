import type { ReactNode } from 'react';

interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  subtitle?: string;
}

/** Centered empty placeholder for lists with no rows. */
export function EmptyState({ icon, title, subtitle }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-border bg-surface-200 text-text-muted">
        {icon}
      </div>
      <p className="text-sm font-medium text-text-secondary">{title}</p>
      {subtitle && <p className="mt-1.5 max-w-xs text-xs text-text-muted">{subtitle}</p>}
    </div>
  );
}

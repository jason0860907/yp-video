import type { ReactNode } from 'react';

interface PageHeaderProps {
  /** Optional muted context line shown on the left. */
  subtitle?: ReactNode;
  /** Page-level toolbar (filters, run/save, …), right-aligned. */
  actions?: ReactNode;
}

/** Page toolbar row. The page title/eyebrow live in the top bar (driven by the
 *  route); this only carries the page's own actions + an optional subtitle. */
export function PageHeader({ subtitle, actions }: PageHeaderProps) {
  if (!subtitle && !actions) return null;
  return (
    <div className="mb-4 flex items-center justify-between gap-4">
      {subtitle ? <p className="text-sm text-text-muted">{subtitle}</p> : <span />}
      {actions && <div className="flex flex-shrink-0 items-center gap-2.5">{actions}</div>}
    </div>
  );
}

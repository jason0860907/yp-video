import type { ReactNode } from 'react';

interface PageHeaderProps {
  /** Small uppercase context label above the title, e.g. "PIPELINE · INGEST". */
  eyebrow?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}

/** Pipeline page header — eyebrow + display title + right-aligned actions,
 *  matching the VolleyIQ prototype's `_pHeader`. */
export function PageHeader({ eyebrow, title, subtitle, actions }: PageHeaderProps) {
  return (
    <div className="mb-5 flex items-end justify-between gap-4">
      <div>
        {eyebrow && (
          <div className="mb-1.5 font-body text-[11px] uppercase tracking-[0.08em] text-text-muted">{eyebrow}</div>
        )}
        <h1 className="font-heading text-[22px] font-bold tracking-tight text-text-primary">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-shrink-0 items-center gap-2.5">{actions}</div>}
    </div>
  );
}

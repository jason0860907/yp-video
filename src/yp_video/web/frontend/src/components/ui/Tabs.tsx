/** Underline tabs sitting on a divider, so a header reads as one line rather
 *  than a box among form controls. A disabled tab can say why in `title`. */

import { cn } from '@/lib/cn';

export interface TabItem<K extends string> {
  key: K;
  label: string;
  disabled?: boolean;
  title?: string;
}

export function Tabs<K extends string>({
  tabs,
  active,
  onChange,
  className,
}: {
  tabs: TabItem<K>[];
  active: K;
  onChange: (key: K) => void;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-wrap items-center gap-1 border-b border-border', className)}>
      {tabs.map((t) => (
        <button
          key={t.key}
          type="button"
          disabled={t.disabled}
          onClick={() => onChange(t.key)}
          title={t.title}
          className={cn(
            '-mb-px border-b-2 px-4 pb-2 pt-1 text-xs font-medium transition-colors',
            t.key === active
              ? 'border-primary text-text-primary'
              : t.disabled
                ? // A muted rose, color alone: distinct from the idle
                  // gray without shouting like a full warning tint.
                  'cursor-not-allowed border-transparent text-red-400/45'
                : 'border-transparent text-text-secondary hover:border-border-light hover:text-text-primary',
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

/** A pick-one switch (mode, kind, tool). The chosen option is brand-green,
 *  like every other selected toggle in the app (timeline zoom, pick mode). */

import { cn } from '@/lib/cn';

export function Segmented<K extends string>({
  options,
  value,
  onChange,
}: {
  options: { key: K; label: string }[];
  value: K;
  onChange: (key: K) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
      {options.map((o) => (
        <button
          key={o.key}
          type="button"
          aria-pressed={o.key === value}
          onClick={() => onChange(o.key)}
          className={cn(
            'rounded-md px-3 py-1 text-xs font-medium transition-colors',
            o.key === value ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

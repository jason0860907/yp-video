/** A pick-one switch (mode, kind, tool). The chosen option reads raised, not
 *  brand-green: green is reserved for the page's one call to action, and a
 *  switch beside it must not compete for that. */

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
            o.key === value
              ? 'bg-surface-300 text-text-primary shadow-sm ring-1 ring-border-light'
              : 'text-text-muted hover:text-text-secondary',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

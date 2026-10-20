/** The App's segmented control (SwiftUI's .segmented picker in dark mode):
 *  a grey track with the chosen segment as a lighter raised pill — not the
 *  pipeline's green toggle. `fill` stretches it into equal segments, the way
 *  the feed's Rally | Action | Score spans its column. */

import { cn } from '@/lib/cn';

export function AppSegmented<K extends string>({
  options,
  value,
  onChange,
  fill,
}: {
  options: { key: K; label: string }[];
  value: K;
  onChange: (key: K) => void;
  fill?: boolean;
}) {
  return (
    <div
      className={cn('gap-0.5 rounded-[9px] bg-ink/[0.08] p-0.5', fill ? 'grid' : 'inline-flex')}
      style={fill ? { gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` } : undefined}
    >
      {options.map((o) => (
        <button
          key={o.key}
          type="button"
          aria-pressed={o.key === value}
          onClick={() => onChange(o.key)}
          className={cn(
            'rounded-[7px] px-3.5 py-1 text-[13px] font-medium text-text-primary transition-colors',
            o.key === value ? 'bg-ink/[0.22] shadow-sm' : 'hover:bg-ink/[0.06]',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

import { cn } from '@/lib/cn';

/** null = any · true = must have · false = must NOT have. */
export type TriState = boolean | null;

export const cycleTri = (v: TriState): TriState => (v === null ? true : v === true ? false : null);

interface TriStateFilterProps<K extends string> {
  fields: ReadonlyArray<{ key: K; label: string }>;
  value: Record<K, TriState>;
  onChange: (key: K, next: TriState) => void;
}

/** Row of cycling property chips — composable AND filters over a list. */
export function TriStateFilter<K extends string>({ fields, value, onChange }: TriStateFilterProps<K>) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {fields.map((f) => {
        const v = value[f.key];
        return (
          <button
            key={f.key}
            type="button"
            onClick={() => onChange(f.key, cycleTri(v))}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium transition-colors',
              v === true && 'border-primary/30 bg-primary/10 text-primary-light',
              v === false && 'border-red-500/30 bg-red-500/10 text-red-300',
              v === null && 'border-border bg-surface-50 text-text-muted hover:text-text-secondary',
            )}
            title="any → has → hasn't"
          >
            <span className="font-mono text-[10px]">{v === true ? '✓' : v === false ? '✕' : '·'}</span>
            {f.label}
          </button>
        );
      })}
    </div>
  );
}

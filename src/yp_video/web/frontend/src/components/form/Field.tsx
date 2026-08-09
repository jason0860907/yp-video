/** The shared input styling and the bare number input. Everything else a
 *  config form needs lives beside this file — schema-bound fields in
 *  SchemaFields.tsx, the labelled wrapper in FieldLabel.tsx.
 */

import { useState } from 'react';
import { cn } from '@/lib/cn';

/** Input/select styling. Exported so pages can apply it to bare inputs. */
export const fieldCls =
  'w-full rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';

/** Number input that never fabricates a value. The raw text lives in a local
 *  draft while editing, so clearing the box doesn't commit `Number('') === 0`;
 *  only parseable input reaches onChange, and blur snaps the display back to
 *  the committed value, clamped into [min, max]. */
export function NumberInput({
  value,
  min,
  max,
  step = 1,
  className,
  onChange,
}: {
  value: number;
  min?: number;
  max?: number;
  step?: number | 'any';
  className?: string;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      type="number"
      value={draft ?? String(value)}
      min={min}
      max={max}
      step={step}
      onFocus={(e) => setDraft(e.target.value)}
      onChange={(e) => {
        setDraft(e.target.value);
        const parsed = e.target.valueAsNumber;
        if (Number.isFinite(parsed)) onChange(parsed);
      }}
      onBlur={() => {
        setDraft(null);
        const clamped = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, value));
        if (clamped !== value) onChange(clamped);
      }}
      className={cn(fieldCls, 'font-mono tabular-nums', className)}
    />
  );
}

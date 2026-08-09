/** Number input over a `number | null` value — null renders as an empty box
 *  with a placeholder, clearing the box commits null (no '' sentinel in the
 *  form state). Same draft-state discipline as NumberInput: only parseable
 *  input reaches onChange, blur clamps a committed number into [min, max].
 */

import { useState } from 'react';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/train/Field';

export function NullableNumberInput({
  value,
  min,
  max,
  step = 1,
  placeholder = 'auto',
  className,
  onChange,
}: {
  value: number | null;
  min?: number;
  max?: number;
  step?: number | 'any';
  placeholder?: string;
  className?: string;
  onChange: (value: number | null) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      type="number"
      value={draft ?? (value === null ? '' : String(value))}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      onFocus={(e) => setDraft(e.target.value)}
      onChange={(e) => {
        setDraft(e.target.value);
        if (e.target.value === '') {
          onChange(null);
          return;
        }
        const parsed = e.target.valueAsNumber;
        if (Number.isFinite(parsed)) onChange(parsed);
      }}
      onBlur={() => {
        setDraft(null);
        if (value === null) return;
        const clamped = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, value));
        if (clamped !== value) onChange(clamped);
      }}
      className={cn(fieldCls, 'font-mono tabular-nums', className)}
    />
  );
}

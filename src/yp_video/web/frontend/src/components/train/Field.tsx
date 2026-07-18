/** Form primitives shared by the train pages' config cards.
 *
 *  Every trainer's config is the same shape — a grid of labelled inputs over
 *  a flat form object mirroring its pydantic request model — so the label
 *  wrapper, the input styling and the enum select live here rather than
 *  being re-typed per page.
 */

import { type ReactNode } from 'react';
import { cn } from '@/lib/cn';

/** Input/select styling. Exported so pages can apply it to bare inputs. */
export const fieldCls =
  'w-full rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';

export function Field({ label, className, children }: { label: string; className?: string; children: ReactNode }) {
  return (
    <label className={cn('block min-w-0 space-y-1', className)}>
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

/** Select over a fixed set of string options (architectures, modes, …). */
export function SelectArch({ value, options, onChange }: { value: string; options: readonly string[]; onChange: (v: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={cn(fieldCls, 'cursor-pointer appearance-none')}>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

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

interface TrainSelectOption {
  value: string;
  label: string;
}

/** Init-checkpoint select, disabled while resuming (weights come from the
 *  run checkpoint instead — the tooltip says so). */
export function InitCheckpointSelect({
  value,
  onChange,
  options,
  resuming,
  emptyLabel = '— From scratch —',
}: {
  value: string;
  onChange: (v: string) => void;
  options: TrainSelectOption[];
  resuming: boolean;
  /** Text of the empty option; null hides it (a trainer that requires a
   *  checkpoint seeds the first one instead of offering from-scratch). */
  emptyLabel?: string | null;
}) {
  return (
    <Field label="Init checkpoint" className="col-span-2">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        title={resuming ? 'Ignored while resuming (weights load from the run checkpoint)' : value}
        disabled={resuming}
        className={cn(fieldCls, 'cursor-pointer appearance-none', resuming && 'opacity-50')}
      >
        {emptyLabel != null && <option value="">{emptyLabel}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

/** Resume-run select — empty means a fresh run. */
export function ResumeRunSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: TrainSelectOption[];
}) {
  return (
    <Field label="Resume from run" className="col-span-3">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        title={value}
        className={cn(fieldCls, 'cursor-pointer appearance-none')}
      >
        <option value="">— New run (train from scratch / init checkpoint) —</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

export type CameraView = 'all' | 'broadcast' | 'sideline';

export function CameraViewSelect({
  value,
  onChange,
}: {
  value: CameraView;
  onChange: (v: CameraView) => void;
}) {
  return (
    <Field label="Camera view">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as CameraView)}
        className={cn(fieldCls, 'cursor-pointer appearance-none')}
      >
        <option value="all">All Views</option>
        <option value="broadcast">Broadcast</option>
        <option value="sideline">Sideline</option>
      </select>
    </Field>
  );
}

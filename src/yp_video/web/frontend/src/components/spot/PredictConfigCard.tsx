import { useEffect, type ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { NumberInput, fieldCls } from '@/components/train/Field';
import type { SpotCheckpoint } from '@/types/api';

export interface NumField<S> {
  key: keyof S & string;
  label: string;
  min: number;
  max?: number;
  step: number;
}

interface BaseSettings {
  checkpoint: string;
  overwrite: boolean;
  stop_vllm: boolean;
}

interface PredictConfigCardProps<S extends BaseSettings> {
  settings: S;
  onChange: (patch: Partial<S>) => void;
  checkpoints: SpotCheckpoint[];
  /** Seeded into settings.checkpoint once, when it is still empty. */
  defaultCheckpoint?: string | null;
  numFields: Array<NumField<S>>;
  /** Extra flavor-specific controls rendered inside the numeric grid. */
  children?: ReactNode;
  overwriteLabel: string;
  runDisabled: boolean;
  onRun: () => void;
}

/** The config card both SPOT predict pages share: checkpoint select, a grid
 *  of numeric fields (each flavor brings its own list), the two standard
 *  checkboxes and the run button. */
export function PredictConfigCard<S extends BaseSettings>({
  settings,
  onChange,
  checkpoints,
  defaultCheckpoint,
  numFields,
  children,
  overwriteLabel,
  runDisabled,
  onRun,
}: PredictConfigCardProps<S>) {
  // Seed checkpoint from the server default once available.
  useEffect(() => {
    if (defaultCheckpoint && !settings.checkpoint) {
      onChange({ checkpoint: defaultCheckpoint } as Partial<S>);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultCheckpoint, settings.checkpoint]);

  return (
    <Card>
      <SectionLabel>Config</SectionLabel>
      <label className="mb-1.5 block text-[10.5px] uppercase tracking-wide text-text-muted">Checkpoint</label>
      <select
        value={settings.checkpoint}
        onChange={(e) => onChange({ checkpoint: e.target.value } as Partial<S>)}
        className={cn(fieldCls, 'mb-3 cursor-pointer appearance-none')}
      >
        {checkpoints.length === 0 && <option value="">No checkpoint</option>}
        {checkpoints.map((c) => (
          <option key={c.path} value={c.path}>
            {c.name} · {c.is_best ? 'best' : `epoch ${c.epoch}`}
            {c.predicts_actor ? ' · fusion' : ''}
          </option>
        ))}
      </select>

      <div className="grid grid-cols-2 gap-2.5">
        {numFields.map((f) => (
          <div key={f.key}>
            <label className="mb-1 block text-[10px] uppercase tracking-wide text-text-muted">{f.label}</label>
            <NumberInput
              value={settings[f.key] as number}
              min={f.min}
              max={f.max}
              step={f.step}
              onChange={(v) => onChange({ [f.key]: v } as Partial<S>)}
            />
          </div>
        ))}
        {children}
      </div>

      <div className="mt-3 space-y-2">
        <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
          <input
            type="checkbox"
            checked={settings.overwrite}
            onChange={(e) => onChange({ overwrite: e.target.checked } as Partial<S>)}
            className="h-3.5 w-3.5 accent-primary"
          />
          {overwriteLabel}
        </label>
        <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
          <input
            type="checkbox"
            checked={settings.stop_vllm}
            onChange={(e) => onChange({ stop_vllm: e.target.checked } as Partial<S>)}
            className="h-3.5 w-3.5 accent-primary"
          />
          Stop vLLM first
        </label>
      </div>

      <Button intent="primary" onClick={onRun} disabled={runDisabled} className="mt-4 w-full">
        Run SPOT
      </Button>
    </Card>
  );
}

/** The amber "SPOT unavailable" banner, rendered only when there is a problem. */
export function SpotProblemBanner({ problem }: { problem: string | null }) {
  if (!problem) return null;
  return (
    <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.06] px-4 py-3 text-sm text-amber-300">
      SPOT unavailable: {problem}
    </div>
  );
}

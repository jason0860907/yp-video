/** The one Source select every multi-store mode shares (rally, action).
 *
 *  Two options, no automatic fallback: the saved Annotation or the machine
 *  Pre-Annotation — what you select is what you see. */

import { fieldCls } from '@/components/form/Field';
import { cn } from '@/lib/cn';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import type { LabelSource, LoadedSource } from './mode';

interface SourceSelectProps {
  source: LabelSource;
  onSource: (s: LabelSource) => void;
  /** What the last load resolved to; rendered right beside the select so
   *  the request and the answer read as one phrase. Null hides the badge. */
  loaded?: LoadedSource | null;
  /** The mode keeps a single store: the select shows it, locked. */
  disabled?: boolean;
}

const LOADED: Record<LoadedSource, { label: string; tone: BadgeTone }> = {
  annotation: { label: 'Annotation', tone: 'brand' },
  'pre-annotation': { label: 'Pre-Annotation', tone: 'warning' },
  none: { label: 'empty', tone: 'neutral' },
};

export function SourceSelect({ source, onSource, loaded, disabled = false }: SourceSelectProps) {
  return (
    <div className="inline-flex items-center gap-2 text-xs text-text-muted">
      <label className="inline-flex items-center gap-2">
        Source
        <select
          value={source}
          onChange={(e) => onSource(e.target.value as LabelSource)}
          disabled={disabled}
          title={disabled ? 'This mode keeps a single store' : undefined}
          className={cn(fieldCls, 'h-9 py-0 text-xs disabled:cursor-not-allowed disabled:opacity-50')}
        >
          <option value="annotation">Annotation</option>
          <option value="pre-annotation">Pre-Annotation</option>
        </select>
      </label>
      {loaded && (
        <span title="The store the editor actually loaded" className="inline-flex items-center gap-1.5">
          loaded
          <Badge tone={LOADED[loaded].tone}>{LOADED[loaded].label}</Badge>
        </span>
      )}
    </div>
  );
}

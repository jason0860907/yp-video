/** The one Source select every multi-store mode shares (rally, action).
 *
 *  Three options both modes understand: Auto (annotation first, machine
 *  output as fallback), the saved Annotation, the machine Pre-Annotation.
 *  Rally's third store — the VLM pass — is the special case, exposed as a
 *  checkbox rather than a fourth option: checked, Pre-Annotation (and
 *  Auto's fallback) reads the VLM pass instead of the SPOT pass. */

import { fieldCls } from '@/components/form/Field';
import { cn } from '@/lib/cn';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import type { LabelSource, LoadedSource } from './mode';

interface SourceSelectProps {
  source: LabelSource;
  onSource: (s: LabelSource) => void;
  vlm: boolean;
  onVlm: (v: boolean) => void;
  /** Rally only — the sole mode with a VLM store. */
  showVlm: boolean;
  /** What the last load resolved to; rendered right beside the select so
   *  the request and the answer read as one phrase. Null hides the badge. */
  loaded?: LoadedSource | null;
}

const LOADED: Record<LoadedSource, { label: string; tone: BadgeTone }> = {
  annotation: { label: 'Annotation', tone: 'brand' },
  'pre-annotation': { label: 'Pre-Annotation', tone: 'warning' },
  vlm: { label: 'VLM', tone: 'warning' },
  none: { label: 'empty', tone: 'neutral' },
};

export function SourceSelect({ source, onSource, vlm, onVlm, showVlm, loaded }: SourceSelectProps) {
  return (
    <div className="inline-flex items-center gap-2 text-xs text-text-muted">
      <label className="inline-flex items-center gap-2">
        Source
        <select value={source} onChange={(e) => onSource(e.target.value as LabelSource)} className={cn(fieldCls, 'h-9 py-0')}>
          <option value="auto">Auto</option>
          <option value="annotation">Annotation</option>
          <option value="pre-annotation">Pre-Annotation</option>
        </select>
      </label>
      {showVlm && (
        <button
          type="button"
          onClick={() => onVlm(!vlm)}
          title="Read the VLM pass instead of the SPOT pass for Pre-Annotation (and Auto's fallback)"
          className={cn(
            'h-9 rounded-lg border px-2.5 font-medium transition-colors',
            vlm
              ? 'border-primary/50 bg-primary/15 text-primary-light'
              : 'border-border-light text-text-muted hover:text-text-secondary',
          )}
        >
          VLM
        </button>
      )}
      {loaded && (
        <span title="The store the editor actually loaded" className="inline-flex items-center gap-1.5">
          loaded
          <Badge tone={LOADED[loaded].tone}>{LOADED[loaded].label}</Badge>
        </span>
      )}
    </div>
  );
}

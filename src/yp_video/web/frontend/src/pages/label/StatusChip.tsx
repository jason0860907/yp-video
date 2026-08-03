/** The shared status chip every mode's picker row renders — one vocabulary,
 *  one color per state, so a video reads the same across all four tabs.
 *  Done keeps the primary tint, Pre-Annotate keeps the amber that machine
 *  output has always worn here; In-Progress gets its own hue so "human is
 *  on it" never blurs into either neighbor. */

import { Badge, type BadgeTone } from '@/components/ui/Badge';
import type { LabelStatus } from '@/lib/labelStatus';

const CHIP: Record<LabelStatus, { label: string; tone: BadgeTone } | null> = {
  unlabeled: null, // untouched rows stay bare — absence reads louder than a gray pill
  'pre-annotate': { label: 'Pre-Annotate', tone: 'warning' },
  'in-progress': { label: 'In-Progress', tone: 'info' },
  done: { label: 'Done', tone: 'success' },
};

export function StatusChip({ status }: { status: LabelStatus }) {
  const chip = CHIP[status];
  if (!chip) return null;
  return <Badge tone={chip.tone}>{chip.label}</Badge>;
}

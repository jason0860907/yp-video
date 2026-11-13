import { Badge } from '@/components/ui/Badge';
import { CUT_KIND_OPTIONS } from '@/lib/cutKinds';
import type { CutKind } from '@/types/api';

/** Camera-view badge — the one visual for a cut kind across pages. */
export function KindBadge({ kind }: { kind: CutKind }) {
  const badge = CUT_KIND_OPTIONS.find((o) => o.kind === kind)?.badge ?? kind;
  const tone = kind === 'broadcast' ? 'info' : kind === 'sideline' ? 'warning' : 'success';
  return <Badge tone={tone}>{badge}</Badge>;
}

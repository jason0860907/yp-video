import { Badge } from '@/components/ui/Badge';
import type { CutKind } from '@/types/api';

/** Camera-view badge — the one visual for broadcast/sideline across pages. */
export function KindBadge({ kind }: { kind: CutKind }) {
  return kind === 'sideline' ? <Badge tone="warning">side</Badge> : <Badge tone="info">cast</Badge>;
}

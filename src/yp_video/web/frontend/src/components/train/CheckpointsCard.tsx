import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import type { AssociationCheckpoint } from '@/types/api';

const pct = (v: number | null | undefined) =>
  v == null ? '—' : `${(v * 100).toFixed(1)}%`;

/** Association-family checkpoint listing, shared by the Association and
 *  Fusion train pages. Legacy fusion actor heads report a different metric
 *  set than independent yp-association runs, hence the two metric lines. */
export function CheckpointsCard({
  title,
  checkpoints,
}: {
  title: string;
  checkpoints: AssociationCheckpoint[];
}) {
  return (
    <Card>
      <SectionLabel>{title}</SectionLabel>
      {checkpoints.length ? (
        <div className="space-y-2">
          {checkpoints.map((checkpoint) => (
            <div
              key={checkpoint.path}
              className="rounded-lg border border-border bg-surface-50 px-3 py-2 text-xs"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-text-primary">{checkpoint.name}</span>
                <Badge tone="success">Association Predict ready</Badge>
                {checkpoint.family === 'legacy-actor-head' ? (
                  <Badge tone="neutral">fusion actor head</Badge>
                ) : null}
                {checkpoint.epoch != null ? (
                  <span className="text-text-muted">epoch {checkpoint.epoch + 1}</span>
                ) : null}
              </div>
              {checkpoint.family === 'legacy-actor-head' ? (
                <div className="mt-1 flex flex-wrap gap-3 font-mono text-[11px] tabular-nums text-text-secondary">
                  <span>overall Top-1 {pct(checkpoint.metrics.all_top1)}</span>
                  <span>hard Top-1 {pct(checkpoint.metrics.hard_top1)}</span>
                  <span>manual Top-1 {pct(checkpoint.metrics.manual_top1)}</span>
                </div>
              ) : (
                <div className="mt-1 flex flex-wrap gap-3 font-mono text-[11px] tabular-nums text-text-secondary">
                  <span>player Top-1 {pct(checkpoint.metrics.player_top1)}</span>
                  <span>overall {pct(checkpoint.metrics.overall_exact)}</span>
                  <span>coverage {pct(checkpoint.metrics.player_coverage)}</span>
                </div>
              )}
              <p className="mt-1 break-all text-[10px] text-text-muted">
                {checkpoint.path}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-text-muted">No checkpoints yet.</p>
      )}
    </Card>
  );
}

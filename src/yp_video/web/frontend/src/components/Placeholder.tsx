import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';

/** Route stub for pages not yet migrated to React. Replaced page-by-page as
 *  the migration proceeds. */
export function Placeholder({ title }: { title: string }) {
  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title={title} subtitle="This page is being migrated to React." />
      <Card label={title}>
        <p className="text-sm text-text-secondary">
          Pending migration. The shared layer (API, SSE, components) is in place — this view will be
          wired up in a later phase.
        </p>
      </Card>
    </div>
  );
}

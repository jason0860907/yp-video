import { Card } from '@/components/ui/Card';

/** Route stub for pages not yet migrated to React. Replaced page-by-page as
 *  the migration proceeds. (The page title is shown in the top bar.) */
export function Placeholder({ title }: { title: string }) {
  return (
    <div className="mx-auto max-w-5xl">
      <Card label={title}>
        <p className="text-sm text-text-secondary">
          Pending migration. The shared layer (API, SSE, components) is in place — this view will be
          wired up in a later phase.
        </p>
      </Card>
    </div>
  );
}

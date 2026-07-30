import { Button } from '@/components/ui/Button';

interface ErrorStateProps {
  title?: string;
  /** The failure detail (usually errMsg(query.error)). */
  message?: string;
  onRetry?: () => void;
}

/** Centered failure placeholder — EmptyState's counterpart, so a broken
 *  fetch never renders as "no data". */
export function ErrorState({ title = 'Failed to load', message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-red-500/20 bg-red-500/10 text-red-400">
        <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
          />
        </svg>
      </div>
      <p className="text-sm font-medium text-red-400">{title}</p>
      {message && <p className="mt-1.5 max-w-sm break-words text-xs text-text-muted">{message}</p>}
      {onRetry && (
        <Button size="sm" className="mt-4" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

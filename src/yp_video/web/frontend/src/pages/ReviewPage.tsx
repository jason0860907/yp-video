import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { PageHeader } from '@/components/ui/PageHeader';
import { AnnotationEditor, type EditorAnnotation, type EditorData } from '@/components/editor/AnnotationEditor';
import { toast } from '@/components/feedback/toast';

interface ReviewResult {
  name: string;
  source: string;
  kind: string;
  subset?: string;
  map?: number;
}
type Filter = 'all' | 'val' | 'train' | 'predict-only' | 'failing' | 'val-failing' | 'broadcast' | 'sideline';

const FILTERS: Array<{ value: Filter; label: string }> = [
  { value: 'all', label: 'All files' },
  { value: 'val', label: 'Validation only' },
  { value: 'train', label: 'Training only' },
  { value: 'predict-only', label: 'Predict only (no annotation)' },
  { value: 'failing', label: 'Failing (mAP < 30%)' },
  { value: 'val-failing', label: 'Val + failing' },
  { value: 'broadcast', label: 'Broadcast only' },
  { value: 'sideline', label: 'Sideline only' },
];

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const streamPath = (vp: string) => apiUrl(API.review.video(vp));

// TAD predictions carry a confidence score; render it as a colored pill.
function scorePill(a: EditorAnnotation) {
  if (a.score == null) return null;
  const cls = a.score > 0.7 ? 'text-emerald-400 bg-emerald-500/10 ring-emerald-500/20' : a.score > 0.4 ? 'text-amber-400 bg-amber-500/10 ring-amber-500/20' : 'text-text-muted bg-ink/5 ring-ink/10';
  return <span className={cn('rounded-full px-2 py-0.5 font-heading text-[10px] font-medium tabular-nums ring-1', cls)}>{(a.score * 100).toFixed(0)}%</span>;
}

export function ReviewPage() {
  const [filter, setFilter] = useState<Filter>('all');
  const [picked, setPicked] = useState('');
  const [data, setData] = useState<EditorData | null>(null);

  const resultsQuery = useQuery({ queryKey: ['review-results'], queryFn: () => apiFetch<ReviewResult[]>(API.review.results) });
  const results = resultsQuery.data ?? [];

  const visible = useMemo(() => {
    const annotatedNames = new Set(results.filter((r) => r.source === 'annotation').map((r) => r.name));
    const passes = (r: ReviewResult) => {
      const isVal = r.subset === 'validation';
      const isFailing = typeof r.map === 'number' && r.map < 0.3;
      const isPredictOnly = r.source === 'tad-prediction' && !annotatedNames.has(r.name);
      switch (filter) {
        case 'val':
          return isVal;
        case 'train':
          return r.subset === 'training';
        case 'predict-only':
          return isPredictOnly;
        case 'failing':
          return isFailing;
        case 'val-failing':
          return isVal && isFailing;
        case 'broadcast':
          return r.kind === 'broadcast';
        case 'sideline':
          return r.kind === 'sideline';
        default:
          return true;
      }
    };
    return results.filter(passes);
  }, [results, filter]);

  const load = async () => {
    if (!picked) return;
    const sep = picked.indexOf('::');
    const source = sep >= 0 ? picked.slice(0, sep) : '';
    const name = sep >= 0 ? picked.slice(sep + 2) : picked;
    try {
      const d = await apiFetch<EditorData>(API.review.result(name, source ? { source } : {}));
      setData(d);
      toast.success(`Loaded ${d.results?.length ?? 0} annotations`);
    } catch (e) {
      toast.error(`Failed to load: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        eyebrow="PIPELINE · RALLY · TAD"
        title="TAD Label"
        actions={
          <>
            <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)} className="cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-xs text-text-primary focus:border-primary/50 focus:outline-none">
              {FILTERS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
            <select value={picked} onChange={(e) => setPicked(e.target.value)} className="max-w-[16rem] cursor-pointer appearance-none truncate rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-xs text-text-primary focus:border-primary/50 focus:outline-none">
              <option value="">Select result file… ({visible.length})</option>
              {visible.map((r) => {
                const mapTag = r.source === 'tad-prediction' && typeof r.map === 'number' ? ` (mAP=${(r.map * 100).toFixed(0)}%)` : '';
                return (
                  <option key={`${r.source}::${r.name}`} value={`${r.source}::${r.name}`}>
                    {r.source === 'annotation' ? '✅' : '🤖'}
                    {r.subset === 'validation' ? ' [VAL]' : ''}
                    {r.kind === 'sideline' ? ' [SIDE]' : ''} {r.name}
                    {mapTag}
                  </option>
                );
              })}
            </select>
            <Button size="sm" intent="primary" onClick={load}>
              Load
            </Button>
          </>
        }
      />

      <AnnotationEditor data={data} saveEndpoint={API.review.annotations} videoStreamPath={streamPath} rowExtras={scorePill} previewBackoff={5} />
    </div>
  );
}

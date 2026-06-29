import { useMemo, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { AnnotationEditor, type EditorAnnotation, type EditorData } from '@/components/editor/AnnotationEditor';
import { toast } from '@/components/feedback/toast';

interface ReviewResult {
  name: string;
  source: string;
  kind: string;
  subset?: string;
  map?: number;
}
type KindFilter = 'all' | 'broadcast' | 'sideline';
type StatusFilter = 'all' | 'pre-labeled' | 'labeled';
type QualityFilter = 'all' | 'val' | 'train' | 'failing' | 'val-failing';

const KIND_FILTERS: Array<{ value: KindFilter; label: string }> = [
  { value: 'all', label: 'All kinds' },
  { value: 'broadcast', label: 'Broadcast' },
  { value: 'sideline', label: 'Sideline' },
];
const STATUS_FILTERS: Array<{ value: StatusFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'pre-labeled', label: 'Pre-labeled' },
  { value: 'labeled', label: 'Labeled' },
];
const QUALITY_FILTERS: Array<{ value: QualityFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'val', label: 'Validation' },
  { value: 'train', label: 'Training' },
  { value: 'failing', label: 'Failing (mAP < 30%)' },
  { value: 'val-failing', label: 'Val + failing' },
];

const fieldCls = 'rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none';
const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const streamPath = (vp: string) => apiUrl(API.review.video(vp));

// TAD predictions carry a confidence score; render it as a colored pill.
function scorePill(a: EditorAnnotation) {
  if (a.score == null) return null;
  const cls = a.score > 0.7 ? 'text-emerald-400 bg-emerald-500/10 ring-emerald-500/20' : a.score > 0.4 ? 'text-amber-400 bg-amber-500/10 ring-amber-500/20' : 'text-text-muted bg-ink/5 ring-ink/10';
  return <span className={cn('rounded-full px-2 py-0.5 font-heading text-[10px] font-medium tabular-nums ring-1', cls)}>{(a.score * 100).toFixed(0)}%</span>;
}

export function ReviewPage() {
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [qualityFilter, setQualityFilter] = useState<QualityFilter>('all');
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

      if (kindFilter !== 'all' && r.kind !== kindFilter) return false;

      if (statusFilter === 'pre-labeled' && !isPredictOnly) return false;
      if (statusFilter === 'labeled' && r.source !== 'annotation') return false;

      switch (qualityFilter) {
        case 'val':
          return isVal;
        case 'train':
          return r.subset === 'training';
        case 'failing':
          return isFailing;
        case 'val-failing':
          return isVal && isFailing;
        default:
          return true;
      }
    };
    return results.filter(passes);
  }, [results, kindFilter, statusFilter, qualityFilter]);

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
      <Card>
        <div className="grid grid-cols-1 items-end gap-3 lg:grid-cols-[8.5rem_8.5rem_8.5rem_minmax(16rem,1fr)_auto]">
          <FieldLabel label="Kind">
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as KindFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              {KIND_FILTERS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </FieldLabel>
          <FieldLabel label="Status">
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as StatusFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              {STATUS_FILTERS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </FieldLabel>
          <FieldLabel label="Quality">
            <select value={qualityFilter} onChange={(e) => setQualityFilter(e.target.value as QualityFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              {QUALITY_FILTERS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </FieldLabel>
          <FieldLabel label="Result file">
            <select value={picked} onChange={(e) => setPicked(e.target.value)} className={cn(fieldCls, 'h-9 w-full truncate py-0')}>
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
          </FieldLabel>
          <Button intent="primary" className="h-9 py-0" onClick={load}>
            Load
          </Button>
        </div>
      </Card>

      <AnnotationEditor data={data} saveEndpoint={API.review.annotations} videoStreamPath={streamPath} rowExtras={scorePill} previewBackoff={5} />
    </div>
  );
}

function FieldLabel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-1.5">
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

import { useMemo, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { copyText } from '@/lib/download';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { AnnotationEditor, type EditorData } from '@/components/editor/AnnotationEditor';
import { toast } from '@/components/feedback/toast';

interface AnnotateResult {
  name: string;
  source: string | string[];
  kind: string;
}
type KindFilter = 'all' | 'broadcast' | 'sideline';
type StatusFilter = 'all' | 'pre-labeled' | 'labeled';

// The list endpoint returns source as an array of {annotation, pre-annotation};
// a file counts as "labeled" once a manual annotation exists for it.
const hasAnnotation = (src: string | string[]) => (Array.isArray(src) ? src : [src]).includes('annotation');

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
// Stable reference so the editor's load effect doesn't re-run each render.
const streamPath = (vp: string) => apiUrl(API.annotate.video(vp));

export function AnnotatePage() {
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [picked, setPicked] = useState('');
  const [data, setData] = useState<EditorData | null>(null);
  const [manifestUrl, setManifestUrl] = useState<string | null>(null);

  const resultsQuery = useQuery({ queryKey: ['annotate-results'], queryFn: () => apiFetch<AnnotateResult[]>(API.annotate.results) });
  const results = resultsQuery.data ?? [];

  const visible = useMemo(
    () =>
      results.filter((r) => {
        if (kindFilter !== 'all' && r.kind !== kindFilter) return false;
        if (statusFilter === 'labeled' && !hasAnnotation(r.source)) return false;
        if (statusFilter === 'pre-labeled' && hasAnnotation(r.source)) return false;
        return true;
      }),
    [results, kindFilter, statusFilter],
  );

  const load = async () => {
    if (!picked) return;
    try {
      const d = await apiFetch<EditorData>(API.annotate.result(picked));
      setData(d);
      setManifestUrl(null);
      toast.success(`Loaded ${d.results?.length ?? 0} annotations`);
    } catch (e) {
      toast.error(`Failed to load: ${errMsg(e)}`);
    }
  };

  // Post-save: push the match to the app's R2 library. Never throws.
  const pushToApp = async (videoName: string) => {
    if (!videoName) return;
    toast.info('Pushing this match to the app…');
    try {
      const res = await apiFetch<{ manifest_url: string; video_uploaded: boolean; rally_count: number }>(API.annotate.publish, {
        method: 'POST',
        body: { video: videoName },
      });
      setManifestUrl(res.manifest_url);
      toast.success(res.video_uploaded ? `Pushed to the app — video + ${res.rally_count} rallies uploaded` : `Pushed to the app — ${res.rally_count} rallies updated`);
    } catch (e) {
      toast.error(`Saved locally, but push to app failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <Card>
        <div className="grid grid-cols-1 items-end gap-3 lg:grid-cols-[8.5rem_8.5rem_minmax(18rem,1fr)_auto]">
          <FieldLabel label="Kind">
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as KindFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              <option value="all">All kinds</option>
              <option value="broadcast">Broadcast</option>
              <option value="sideline">Sideline</option>
            </select>
          </FieldLabel>
          <FieldLabel label="Status">
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as StatusFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              <option value="all">All</option>
              <option value="pre-labeled">Pre-labeled</option>
              <option value="labeled">Labeled</option>
            </select>
          </FieldLabel>
          <FieldLabel label="Result file">
            <select value={picked} onChange={(e) => setPicked(e.target.value)} className={cn(fieldCls, 'h-9 w-full truncate py-0')}>
              <option value="">Select result file… ({visible.length})</option>
              {visible.map((r) => (
                <option key={r.name} value={r.name}>
                  {hasAnnotation(r.source) ? '✅' : '⚡'}
                  {r.kind === 'sideline' ? ' [SIDE]' : ''} {r.name}
                </option>
              ))}
            </select>
          </FieldLabel>
          <Button intent="primary" className="h-9 py-0" onClick={load}>
            Load
          </Button>
        </div>
      </Card>

      <AnnotationEditor data={data} saveEndpoint={API.annotate.annotations} videoStreamPath={streamPath} previewBackoff={3} onSaved={pushToApp} />

      {manifestUrl && (
        <div className="rounded-xl border border-border bg-surface-100 p-4 text-xs">
          <div className="text-text-muted">App import URL — paste into VolleyIQ → Settings → Library manifest URL</div>
          <div className="mt-2 flex items-center gap-2">
            <input readOnly value={manifestUrl} className="min-w-0 flex-1 rounded-lg border border-border-light bg-surface-50 px-2.5 py-1.5 font-mono text-text-secondary" />
            <Button
              size="sm"
              onClick={async () => {
                try {
                  await copyText(manifestUrl);
                  toast.success('Manifest URL copied');
                } catch {
                  toast.error('Copy failed');
                }
              }}
            >
              Copy
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

const fieldCls = 'rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none';

function FieldLabel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-1.5">
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

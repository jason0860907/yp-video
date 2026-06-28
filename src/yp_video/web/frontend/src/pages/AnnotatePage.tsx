import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { copyText } from '@/lib/download';
import { Button } from '@/components/ui/Button';
import { PageHeader } from '@/components/ui/PageHeader';
import { AnnotationEditor, type EditorData } from '@/components/editor/AnnotationEditor';
import { toast } from '@/components/feedback/toast';

interface AnnotateResult {
  name: string;
  source: string | string[];
  kind: string;
}
type KindFilter = 'all' | 'broadcast' | 'sideline';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
// Stable reference so the editor's load effect doesn't re-run each render.
const streamPath = (vp: string) => apiUrl(API.annotate.video(vp));

export function AnnotatePage() {
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [picked, setPicked] = useState('');
  const [data, setData] = useState<EditorData | null>(null);
  const [manifestUrl, setManifestUrl] = useState<string | null>(null);

  const resultsQuery = useQuery({ queryKey: ['annotate-results'], queryFn: () => apiFetch<AnnotateResult[]>(API.annotate.results) });
  const results = resultsQuery.data ?? [];

  const visible = useMemo(() => results.filter((r) => kindFilter === 'all' || r.kind === kindFilter), [results, kindFilter]);

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
      <PageHeader
        actions={
          <>
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as KindFilter)} className="cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-xs text-text-primary focus:border-primary/50 focus:outline-none">
              <option value="all">All kinds</option>
              <option value="broadcast">Broadcast only</option>
              <option value="sideline">Sideline only</option>
            </select>
            <select value={picked} onChange={(e) => setPicked(e.target.value)} className="max-w-[16rem] cursor-pointer appearance-none truncate rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-xs text-text-primary focus:border-primary/50 focus:outline-none">
              <option value="">Select result file… ({visible.length})</option>
              {visible.map((r) => (
                <option key={r.name} value={r.name}>
                  {String(r.source).includes('annotation') ? '✅' : '⚡'}
                  {r.kind === 'sideline' ? ' [SIDE]' : ''} {r.name}
                </option>
              ))}
            </select>
            <Button size="sm" intent="primary" onClick={load}>
              Load
            </Button>
          </>
        }
      />

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

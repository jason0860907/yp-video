/** Rally Label panel: edit one video's rally spans and push them to the app.
 *
 *  The panel is keyed by the cut filename (`video` prop) — the annotation
 *  file it loads is `<stem>_annotations.jsonl`, the same mapping the backend
 *  applies (`find_cut(f"{stem}.mp4")` in routers/annotate.py). "Which store
 *  to read" is a per-load choice, not a list filter, so the shared Source
 *  select (SourceSelect) sits beside the mode tabs rather than in the picker
 *  row; the page owns its state and passes it down.
 *
 *  No dirty guard: the editor autosaves 2 s after editing stops and flushes
 *  on pagehide, so leaving loses at most a couple of seconds of work.
 */

import { useEffect, useState } from 'react';
import { API, ApiError, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { copyText } from '@/lib/download';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { AnnotationEditor, type EditorData } from '@/components/editor/AnnotationEditor';
import { toast } from '@/components/feedback/toast';
import { rallyStatus } from '@/lib/labelStatus';
import { STATUS_OPTIONS, type LabelSource, type LoadedSource, type ModeDescriptor, type PlaybackClock } from './mode';

/** Cut filename → its rally annotation file. */
const rallyResultName = (video: string) => `${video.replace(/\.[^.]+$/, '')}_annotations.jsonl`;

export const RALLY_MODE: ModeDescriptor = {
  key: 'rally',
  label: 'Rally',
  statusOptions: STATUS_OPTIONS,
  status: rallyStatus,
  matches: (row, status) => status === 'all' || rallyStatus(row) === status,
  available: (row) => Boolean(row.rally),
  hint: () => 'No rally annotation or prediction for this video yet',
  // The annotation lives in R2 with no local cut listed — worth flagging
  // here because only this tab can open such a row.
  rowExtras: (row) => (row.rallyOnly ? <Badge tone="warning">rally only</Badge> : null),
  doneApi: (video) => API.annotate.done(video),
  listKey: 'annotate-results',
  hasSources: true,
};

// Stable reference so the editor's load effect doesn't re-run each render.
const streamPath = (vp: string) => apiUrl(API.annotate.video(vp));

/** The shared (source, vlm) choice → this store's tag — exactly one store,
 *  no fallback. Rally keeps three stores; the backend tags are annotation /
 *  spot-pre-annotation (SPOT) / pre-annotation (VLM), and the VLM checkbox
 *  redirects Pre-Annotation to the VLM pass. */
const loadRally = (name: string, source: LabelSource, vlm: boolean): Promise<EditorData> =>
  apiFetch(API.annotate.result(name, {
    source: source === 'annotation' ? 'annotation' : vlm ? 'pre-annotation' : 'spot-pre-annotation',
  }));

/** Backend store tag → the shared LoadedSource vocabulary. The rally tag
 *  'pre-annotation' is the VLM pass; SPOT wears 'spot-pre-annotation'. */
const loadedFromTag = (tag: unknown): LoadedSource =>
  tag === 'annotation' ? 'annotation' : tag === 'spot-pre-annotation' ? 'pre-annotation' : tag === 'pre-annotation' ? 'vlm' : 'none';

export function RallyPanel({ video, source, vlm, onLoaded, onSaved, clock }: { video: string; source: LabelSource; vlm: boolean; onLoaded?: (s: LoadedSource) => void; onSaved?: () => void; clock?: PlaybackClock }) {
  const [data, setData] = useState<EditorData | null>(null);
  const [manifestUrl, setManifestUrl] = useState<string | null>(null);

  // Load on video pick and on Source change — the picked file is already
  // open, so a Source switch re-reads it from the newly chosen store.
  useEffect(() => {
    if (!video) {
      setData(null);
      setManifestUrl(null);
      return;
    }
    let stale = false;
    void (async () => {
      try {
        const d = await loadRally(rallyResultName(video), source, vlm);
        if (stale) return;
        setData(d);
        setManifestUrl(null);
        onLoaded?.(loadedFromTag(d.source));
        toast.success(`Loaded ${d.results?.length ?? 0} annotations (${String(d.source || '')})`);
      } catch (e) {
        if (stale) return;
        onLoaded?.('none');
        // The selected store simply has no file yet — an empty editor, not
        // an error: rally spans can be drawn from scratch.
        if (e instanceof ApiError && e.status === 404) {
          setData({ video, results: [] });
          setManifestUrl(null);
        } else {
          toast.error(`Failed to load: ${errMsg(e)}`);
        }
      }
    })();
    return () => {
      stale = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video, source, vlm]);

  // Post-save: push the match to the app's R2 library. Never throws.
  const pushToApp = async (videoName: string) => {
    if (!videoName) return;
    // The save just wrote the annotation store — whatever was loaded before,
    // that is what the editor is showing now.
    onLoaded?.('annotation');
    onSaved?.();
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
    <div className="space-y-5">
      <AnnotationEditor
        data={data}
        saveEndpoint={API.annotate.annotations}
        videoStreamPath={streamPath}
        previewBackoff={3}
        onSaved={pushToApp}
        initialTime={() => clock?.read(video) ?? null}
        onTimeChange={(t) => clock?.write(video, t)}
      />

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

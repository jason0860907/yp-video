/** Rally Label panel: edit one video's rally spans.
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
import { Badge } from '@/components/ui/Badge';
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

export function RallyPanel({ video, source, vlm, onLoaded, onSaved, clock, clipsOpen, onClipsClose }: { video: string; source: LabelSource; vlm: boolean; onLoaded?: (s: LoadedSource) => void; onSaved?: () => void; clock?: PlaybackClock; clipsOpen: boolean; onClipsClose: () => void }) {
  const [data, setData] = useState<EditorData | null>(null);

  // Load on video pick and on Source change — the picked file is already
  // open, so a Source switch re-reads it from the newly chosen store.
  useEffect(() => {
    if (!video) {
      setData(null);
      return;
    }
    let stale = false;
    void (async () => {
      try {
        const d = await loadRally(rallyResultName(video), source, vlm);
        if (stale) return;
        setData(d);
        onLoaded?.(loadedFromTag(d.source));
        toast.success(`Loaded ${d.results?.length ?? 0} annotations (${String(d.source || '')})`);
      } catch (e) {
        if (stale) return;
        onLoaded?.('none');
        // The selected store simply has no file yet — an empty editor, not
        // an error: rally spans can be drawn from scratch.
        if (e instanceof ApiError && e.status === 404) {
          setData({ video, results: [] });
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

  // The save just wrote the annotation store — whatever was loaded before,
  // that is what the editor is showing now.
  const afterSave = () => {
    onLoaded?.('annotation');
    onSaved?.();
  };

  return (
    <div className="space-y-5">
      <AnnotationEditor
        data={data}
        saveEndpoint={API.annotate.annotations}
        videoStreamPath={streamPath}
        previewBackoff={3}
        onSaved={afterSave}
        initialTime={() => clock?.read(video) ?? null}
        onTimeChange={(t) => clock?.write(video, t)}
        clipsOpen={clipsOpen}
        onClipsClose={onClipsClose}
      />
    </div>
  );
}

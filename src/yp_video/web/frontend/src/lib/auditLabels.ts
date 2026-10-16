/**
 * Display names for audit actions.
 *
 * The backend records the FastAPI route template verbatim, so a new endpoint
 * is audited the day it is written without anyone touching this file. This map
 * only makes the common ones readable; anything unmapped falls through to the
 * raw "METHOD /api/..." string, so a stale entry here can never lose an event.
 */

const ACTION_LABELS: Record<string, string> = {
  // Labeling — the rows there are most of the trail. One name per panel of
  // the Label page, phrased the same way, because the reader is scanning a
  // column: "儲存 X 標註" for the four that record work, distinct wording for
  // the one-off actions beside them.
  'POST /api/annotate/annotations': 'Save Rally labels',
  'POST /api/action-annotate/annotations': 'Save Action labels',
  'POST /api/actor-association/fix/{name}': 'Save Association labels',
  'PUT /api/detection-label/frame/{name}': 'Save Detection labels',
  'PUT /api/court/video/{name}': 'Save Court calibration',
  'PUT /api/reid/players/{name}': 'Save ReID labels',
  'PUT /api/annotate/done/{name}': 'Rally labels done',
  'PUT /api/action-annotate/done/{name:path}': 'Action labels done',
  'PUT /api/actor-association/done/{name:path}': 'Association labels done',
  'PUT /api/reid/done/{name}': 'ReID labels done',
  'POST /api/actor-association/confirm/{name}': 'Confirm auto actors',
  'POST /api/reid/seed-cluster/{name}': 'Recluster',

  // Video preparation.
  'POST /api/download/start': 'Download YouTube video',
  'POST /api/download/{session_id}/cancel': 'Cancel download',
  'POST /api/cut/export': 'Cut into sets',
  'DELETE /api/cut/video/{name}': 'Delete source video',
  'POST /api/annotate/clip': 'Download clip',
  'POST /api/annotate/clip-zip': 'Download clip zip',

  // Training and inference.
  'POST /api/inference/start': 'Start Inference',
  'POST /api/fusion-model/train': 'Start Train',
  'POST /api/reid-train/train': 'Start ReID Train',
  'POST /api/reid-train/export': 'Export ReID dataset',
  'POST /api/reid/embed': 'Run ReID embedding',
  'POST /api/extraction/detect': 'Run Player Detection',
  'POST /api/tracklets/run': 'Run Rally Tracking',

  // Storage and system — the destructive end of the surface.
  'POST /api/upload/start': 'Upload to R2',
  'POST /api/upload/download': 'Download from R2',
  'POST /api/upload/delete-local': 'Delete local files',
  'POST /api/upload/delete-r2': 'Delete R2 objects',
  'POST /api/jobs/{job_id}/cancel': 'Cancel job',

  // Background job lifecycle.
  'job.running': 'Job started',
  'job.completed': 'Job completed',
  'job.failed': 'Job failed',
  'job.cancelled': 'Job cancelled',
};

export function actionLabel(action: string): string {
  return ACTION_LABELS[action] ?? action;
}

/** True for the job lifecycle rows, which the UI tints differently. */
export const isJobAction = (action: string): boolean => action.startsWith('job.');

/** "rallies=37 · files=4" — the numbers a route chose to record. */
export function summaryText(summary: Record<string, unknown>): string {
  return Object.entries(summary)
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(' · ');
}

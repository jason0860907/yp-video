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
  'POST /api/annotate/annotations': '儲存 Rally 標註',
  'POST /api/action-annotate/annotations': '儲存 Action 標註',
  'POST /api/actor-association/fix/{name}': '儲存 Association 標註',
  'PUT /api/reid/players/{name}': '儲存 ReID 標註',
  'PUT /api/annotate/done/{name}': 'Rally 標註完成',
  'PUT /api/action-annotate/done/{name:path}': 'Action 標註完成',
  'PUT /api/actor-association/done/{name:path}': 'Association 標註完成',
  'PUT /api/reid/done/{name}': 'ReID 標註完成',
  'POST /api/actor-association/confirm/{name}': '批次確認自動歸屬',
  'POST /api/reid/seed-cluster/{name}': '重算分群',

  // Video preparation.
  'POST /api/download/start': '下載 YouTube 影片',
  'POST /api/download/{session_id}/cancel': '取消下載',
  'POST /api/cut/export': '切分 set',
  'DELETE /api/cut/video/{name}': '刪除原始影片',
  'POST /api/detect/start': '啟動 VLM 偵測',
  'POST /api/detect/convert': 'VLM 片段轉 Rally',
  'POST /api/annotate/clip': '下載單一片段',
  'POST /api/annotate/clip-zip': '下載片段壓縮檔',
  'POST /api/annotate/publish': '發佈到 iOS app',

  // Training and inference.
  'POST /api/spot-train/start': '啟動 Rally SPOT 訓練',
  'POST /api/spot-predict/start': '啟動 Rally SPOT 推論',
  'POST /api/action-train/start': '啟動 Action 訓練',
  'POST /api/action-annotate/prelabel-batch': '批次 Action 預標',
  'POST /api/actor-association/train': '啟動 Association 訓練',
  'POST /api/actor-association/predict': '批次重算動作歸屬',
  'POST /api/fusion-model/train': '啟動 Fusion 訓練',
  'POST /api/reid-train/train': '啟動 ReID 訓練',
  'POST /api/reid-train/export': '匯出 ReID 資料集',
  'POST /api/reid/embed': '計算 ReID 特徵',
  'POST /api/extraction/detect': '偵測球員',
  'POST /api/tracklets/run': '追蹤球員',

  // Storage and system — the destructive end of the surface.
  'POST /api/upload/start': '上傳到 R2',
  'POST /api/upload/download': '從 R2 下載',
  'POST /api/upload/delete-local': '刪除本機檔案',
  'POST /api/upload/delete-r2': '刪除 R2 物件',
  'POST /api/jobs/{job_id}/cancel': '取消工作',
  'POST /api/system/vllm/start': '啟動 vLLM',
  'POST /api/system/vllm/stop': '關閉 vLLM',

  // Background job lifecycle.
  'job.running': '工作開始執行',
  'job.completed': '工作完成',
  'job.failed': '工作失敗',
  'job.cancelled': '工作已取消',
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

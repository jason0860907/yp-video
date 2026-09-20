import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { fieldCls } from '@/components/form/Field';
import { Button } from '@/components/ui/Button';
import { FeedbackPanel } from './FeedbackPanel';
import { ResultPreview } from './ResultPreview';
import type { Bundle, Preview, Review, ReviewSummary, WindowMode } from './types';

export function AppReviewPage() {
  const queryClient = useQueryClient();
  const [source, setSource] = useState<'local' | 'artifacts' | 'cloud' | 'queue'>('local');
  const [video, setVideo] = useState('');
  const [mode, setMode] = useState<WindowMode>('full_play');
  const [corrected, setCorrected] = useState(true);
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [review, setReview] = useState<Review | null>(null);
  const [fileEpoch, setFileEpoch] = useState(0);
  const [files, setFiles] = useState<Record<string, unknown>>({});
  const [cloudMatch, setCloudMatch] = useState('');
  const [job, setJob] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [pipelineMedia, setPipelineMedia] = useState(false);
  const videos = useQuery({
    queryKey: ['app-review-videos'],
    queryFn: () => apiFetch<{ name: string }[]>(API.appReview.videos),
  });
  const local = useQuery({
    queryKey: ['app-review-local', video, mode],
    placeholderData: (previous, query) => (query?.queryKey[1] === video ? previous : undefined),
    enabled: source === 'local' && !!video,
    queryFn: () => apiFetch<{ preview: Preview }>(API.appReview.local(video, mode)),
  });
  const preview = useQuery({
    queryKey: ['app-review-preview', review?.id ?? bundle, mode, corrected],
    placeholderData: (previous, query) =>
      query?.queryKey[1] === (review?.id ?? bundle) ? previous : undefined,
    enabled: source !== 'local' && !!bundle,
    queryFn: () =>
      apiFetch<Preview>(API.appReview.preview, {
        method: 'POST',
        body: { bundle, mode, corrected },
      }),
  });
  const queue = useQuery({
    queryKey: ['app-review-queue'],
    queryFn: () => apiFetch<ReviewSummary[]>(API.appReview.reviews),
  });
  const cloud = useQuery({
    queryKey: ['app-review-cloud'],
    enabled: source === 'cloud',
    queryFn: () =>
      apiFetch<{ user_id: string; match_id: string; updated_at: string }[]>(API.appReview.cloud),
  });
  const [user = '', match = ''] = cloudMatch.split('/');
  const results = useQuery({
    queryKey: ['app-review-cloud-results', user, match],
    enabled: source === 'cloud' && !!match,
    queryFn: () =>
      apiFetch<{ key: string; last_modified: string }[]>(API.appReview.cloudResults(user, match)),
  });
  const loadReview = (r: Review) => {
    setReview(r);
    setBundle(r.bundle);
    void queryClient.invalidateQueries({ queryKey: ['app-review-queue'] });
  };
  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  const importFiles = () =>
    run(async () => {
      if (!files.result) throw new Error('請選擇原始 analysis result JSON');
      const next = {
        result: files.result,
        corrections: files.corrections ?? null,
        identification: files.identification ?? null,
        library_rallies: files.library_rallies ?? null,
      } as Bundle;
      if (files.corrections || files.library_rallies)
        loadReview(await apiFetch<Review>(API.appReview.reviews, { method: 'POST', body: next }));
      else {
        await apiFetch(API.appReview.preview, { method: 'POST', body: { bundle: next } });
        setBundle(next);
        setReview(null);
        setPipelineMedia(true);
      }
    });
  const data = source === 'local' ? local.data?.preview : preview.data;
  const media =
    source === 'local' || pipelineMedia
      ? video
        ? apiUrl(`/action-annotate/video/${encodeURIComponent(video)}`)
        : ''
      : review
        ? apiUrl(API.appReview.media(review.id))
        : '';
  const failure =
    error ||
    (videos.error && errMsg(videos.error)) ||
    (source === 'local' && local.error && errMsg(local.error)) ||
    (source !== 'local' && preview.error && errMsg(preview.error)) ||
    (source === 'cloud' &&
      (cloud.error || results.error) &&
      errMsg(cloud.error || results.error)) ||
    (source === 'queue' && queue.error && errMsg(queue.error));
  return (
    <div className="mx-auto max-w-7xl space-y-5 p-5 text-text-primary">
      <div>
        <h1 className="text-2xl font-semibold">App 結果預覽與修正審核</h1>
        <p className="mt-2 text-sm text-text-secondary">
          用 App 的 Rally／Action／Score 規則回看片段，將使用者修正交由人工審核。
        </p>
      </div>
      <div className="flex flex-wrap gap-2">
        {(
          [
            ['local', 'Pipeline 結果'],
            ['cloud', '雲端修正'],
            ['artifacts', '匯入 JSON'],
            ['queue', '審核紀錄'],
          ] as const
        ).map(([key, label]) => (
          <Button
            key={key}
            disabled={busy}
            intent={source === key ? 'primary' : 'default'}
            onClick={() => {
              setSource(key);
              setError('');
              setReview(null);
              setBundle(null);
            }}
          >
            {label}
          </Button>
        ))}
      </div>
      <label className="block text-sm">
        {source === 'local' ? '預覽影片' : '對應的 pipeline 影片（匯入標註時必選）'}
        <select
          aria-label="Pipeline 影片"
          className={fieldCls}
          value={video}
          onChange={(e) => setVideo(e.target.value)}
        >
          <option value="">選擇影片</option>
          {videos.data?.map((v) => (
            <option key={v.name} value={v.name}>
              {v.name}
            </option>
          ))}
        </select>
      </label>
      {source === 'artifacts' && (
        <div className="space-y-3 rounded border border-border-light p-4">
          <p className="text-sm">
            使用目前格式的原始分析結果、corrections 6.0，以及選用的 identify v4。Library Rally
            快照為 /sync 中這場比賽的完整 rallies 陣列，包含 UUID
            與使用者裁切；未提供時會標示無法還原的部分。
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            {(
              [
                ['result', 'Analysis result（必填）'],
                ['corrections', 'Corrections'],
                ['identification', 'Identify result'],
                ['library_rallies', 'Library Rally 快照'],
              ] as const
            ).map(([key, label]) => (
              <label key={`${key}:${fileEpoch}`} className="text-sm">
                {label}
                <input
                  className={fieldCls}
                  type="file"
                  accept=".json,application/json"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file)
                      void run(async () => {
                        const value: unknown = JSON.parse(await file.text());
                        setFiles((old) => ({ ...old, [key]: value }));
                      });
                  }}
                />
              </label>
            ))}
          </div>
          <Button disabled={busy || !files.result} onClick={() => void importFiles()}>
            載入預覽與審核
          </Button>
          <Button
            disabled={busy}
            onClick={() => {
              setFiles({});
              setFileEpoch((n) => n + 1);
              setBundle(null);
              setReview(null);
            }}
          >
            清除已載入資料
          </Button>
        </div>
      )}
      {source === 'cloud' && (
        <div className="space-y-3">
          <select
            aria-label="雲端比賽"
            className={fieldCls}
            value={cloudMatch}
            onChange={(e) => {
              setCloudMatch(e.target.value);
              setJob('');
            }}
          >
            <option value="">選擇有修正的比賽</option>
            {cloud.data?.map((c) => (
              <option key={`${c.user_id}/${c.match_id}`} value={`${c.user_id}/${c.match_id}`}>
                {c.match_id} · {c.updated_at} · {c.user_id}
              </option>
            ))}
          </select>
          <select
            aria-label="分析版本"
            className={fieldCls}
            value={job}
            onChange={(e) => setJob(e.target.value)}
          >
            <option value="">選擇要核對的分析版本</option>
            {results.data?.map((r) => (
              <option key={r.key} value={r.key.split('/').at(-1)!.slice(0, -5)}>
                {r.key.split('/').at(-1)} · {r.last_modified}
              </option>
            ))}
          </select>
          <p className="text-sm text-text-muted">
            修正快照未記錄 analysis job ID，請選擇要對照的版本；未對應的修正會列出原因。
          </p>
          <Button
            disabled={busy || !job}
            onClick={() =>
              void run(async () =>
                loadReview(
                  await apiFetch<Review>(API.appReview.cloudImport, {
                    method: 'POST',
                    body: { user, match, job },
                  }),
                ),
              )
            }
          >
            讀取雲端快照
          </Button>
        </div>
      )}
      {source === 'queue' && (
        <div className="max-h-48 space-y-2 overflow-auto">
          {queue.data?.length === 0 && <p>尚無審核紀錄。</p>}
          {queue.data?.map((r) => (
            <button
              key={r.id}
              className="block w-full rounded border border-border-light p-3 text-left text-sm"
              onClick={() =>
                void run(async () => loadReview(await apiFetch<Review>(API.appReview.review(r.id))))
              }
            >
              {r.match_id} · {r.job_id} · 已審核 {r.reviewed}/{r.total}
            </button>
          ))}
        </div>
      )}
      {failure && (
        <p role="alert" className="rounded border border-red-500/30 p-3 text-sm text-red-400">
          {failure}
        </p>
      )}
      {(busy || (source === 'local' ? local.isFetching : preview.isFetching)) && (
        <p role="status">載入中…</p>
      )}
      {data && (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <select
              aria-label="片段取景"
              className={`${fieldCls} max-w-xs`}
              value={mode}
              onChange={(e) => setMode(e.target.value as WindowMode)}
            >
              <option value="full_play">組織進攻</option>
              <option value="to_next">單一動作</option>
              <option value="whole_rally">整個 Rally</option>
            </select>
            {source !== 'local' && (
              <>
                <label className="flex gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={corrected}
                    onChange={(e) => setCorrected(e.target.checked)}
                  />
                  套用 App 修正
                </label>
                <label className="flex gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={pipelineMedia}
                    onChange={(e) => setPipelineMedia(e.target.checked)}
                  />
                  播放對應的 pipeline 影片
                </label>
              </>
            )}
          </div>
          <ResultPreview
            key={`${source}:${review?.id ?? bundle?.result.job_id ?? video}`}
            data={data}
            media={media}
            local={source === 'local'}
          />
          {source !== 'local' && review && (
            <FeedbackPanel
              key={`${review.id}:${video}`}
              review={review}
              video={video}
              media={media}
              onChange={loadReview}
            />
          )}
        </>
      )}
    </div>
  );
}

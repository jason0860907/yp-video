/** Starting Rally Tracking, from wherever the lack of tracklets is felt.
 *
 *  Tracking stopped being a step inside extraction when it stopped reading
 *  the action annotation — it needs rally spans and nothing else, so it can
 *  run while actions are still being labeled. Two pages now hit that
 *  prerequisite (ReID Predict needs tracklets to extract, Association Predict
 *  needs them to rank), and neither should own the button: the gate and the
 *  POST live here so the two cannot drift apart.
 */
import { useMemo } from 'react';
import { API, ApiError, apiFetch } from '@/lib/api';
import { STAGE_HINT } from '@/components/video/PipelineChips';
import { toast } from '@/components/feedback/toast';
import type { Job, PipelineState } from '@/types/api';

const errMsg = (e: unknown) =>
  e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e);

/** The job type to keep in a page's job list when it offers this button. */
export const TRACKING_JOB_TYPE = 'player_tracking';

interface Options {
  /** Any video list carrying a name and its pipeline state. */
  videos: readonly { name: string; pipeline: PipelineState }[];
  selected: ReadonlySet<string>;
  onJob: (job: Job) => void;
  overwrite?: boolean;
  stopVllm?: boolean;
}

export function useRallyTracking({
  videos,
  selected,
  onJob,
  overwrite = false,
  stopVllm = false,
}: Options) {
  const blocked = useMemo(() => {
    const chosen = videos.filter((v) => selected.has(v.name));
    return chosen.some((v) => v.pipeline.rally_sources.length === 0)
      ? STAGE_HINT.rallies
      : null;
  }, [videos, selected]);

  /** Videos in the selection that would actually gain something. */
  const missing = useMemo(
    () =>
      videos.filter((v) => selected.has(v.name) && !v.pipeline.has_tracks).length,
    [videos, selected],
  );

  const run = async (): Promise<boolean> => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return false;
    }
    try {
      const job = await apiFetch<Job>(API.reid.track, {
        method: 'POST',
        body: { videos: names, overwrite, stop_vllm: stopVllm },
      });
      onJob(job);
      toast.success(`Started Rally Tracking for ${names.length} video(s)`);
      return true;
    } catch (e) {
      toast.error(`Rally Tracking start failed: ${errMsg(e)}`);
      return false;
    }
  };

  return { run, blocked, missing };
}

/** Analysis: the play reconstructed on a calibrated court — positions,
 *  flights, the rally in 3D. Only videos whose court has been marked are
 *  listed; calibrating them is Label → Court. ?video= keeps the pick
 *  shareable and refresh-safe. */

import { useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useUnionVideos } from '@/lib/useUnionVideos';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { CopyFilenameButton } from '@/components/video/CopyFilenameButton';
import { KindBadge } from '@/components/video/KindBadge';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { StatusChip } from '@/pages/label/StatusChip';
import { AnalysisPanel } from './AnalysisPanel';

export function AnalysisPage() {
  const [params, setParams] = useSearchParams();
  const video = params.get('video') ?? '';
  const { videos, query } = useUnionVideos();
  const calibrated = useMemo(
    () => videos.filter((v) => v.court && v.court.status !== 'unlabeled'),
    [videos],
  );
  const pick = (name: string) =>
    setParams(
      (prev) => {
        const out = new URLSearchParams(prev);
        if (name) out.set('video', name);
        else out.delete('video');
        return out;
      },
      { replace: true },
    );

  return (
    <div className="space-y-5">
      <Card>
        <div className="flex flex-wrap items-center gap-2">
          <VideoCombobox
            className="min-w-[20rem] flex-1"
            items={calibrated}
            value={video}
            onChange={pick}
            query={query}
            placeholder={`Search ${calibrated.length} calibrated videos…`}
            renderItem={(v) => (
              <>
                <KindBadge kind={v.kind} />
                <span className="min-w-0 flex-1 break-all font-mono">{v.name}</span>
                <StatusChip status={v.court?.status ?? 'unlabeled'} />
              </>
            )}
          />
          {video && <CopyFilenameButton name={video} />}
          {video && (
            <Link
              to={`/label?mode=court&video=${encodeURIComponent(video)}`}
              className="text-[11px] text-text-muted hover:text-text-primary"
            >
              Calibrate in Label →
            </Link>
          )}
        </div>
      </Card>
      {video ? (
        <AnalysisPanel key={video} video={video} />
      ) : (
        <Card>
          <EmptyState
            icon="◎"
            title="Pick a calibrated video"
            subtitle="Videos appear here once their court has been marked in Label → Court."
          />
        </Card>
      )}
    </div>
  );
}

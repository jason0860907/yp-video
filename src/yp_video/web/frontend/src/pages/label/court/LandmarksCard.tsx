/** The calibration card: which landmarks are marked, the net height, and
 *  how well the marks agree (floor fit and camera solve). */

import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { LandmarkGuide } from './LandmarkGuide';
import { LANDMARK_LABELS, type CourtState } from './geometry';

const NET_HEIGHTS = [
  { value: 2.43, label: 'Men · 2.43 m' },
  { value: 2.24, label: 'Women · 2.24 m' },
];

export function LandmarksCard({
  state,
  armed,
  onArm,
  onRemove,
  onNetHeight,
  frameHeight,
}: {
  state: CourtState;
  armed: string | null;
  onArm: (name: string) => void;
  onRemove: (name: string) => void;
  onNetHeight: (metres: number) => void;
  /** Pixels per frame height, to state the reprojection error in pixels. */
  frameHeight: number | null;
}) {
  const { fit, camera } = state;
  return (
    <>
      <SectionLabel>Landmarks · {Object.keys(state.points).length} marked</SectionLabel>
      <LandmarkGuide state={state} armed={armed} onArm={onArm} />
      <div className="grid grid-cols-2 gap-1.5">
        {[...Object.keys(state.landmarks), ...Object.keys(state.net_landmarks)].map((name) => {
          const isMarked = name in state.points;
          return (
            <div key={name} className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onArm(name)}
                className={cn(
                  'min-w-0 flex-1 truncate rounded-md border px-2 py-1 text-left text-[11px] transition-colors',
                  armed === name
                    ? 'border-primary text-text-primary'
                    : isMarked
                      ? 'border-yellow-400/40 text-text-secondary'
                      : 'border-border text-text-muted hover:text-text-primary',
                )}
              >
                {LANDMARK_LABELS[name] ?? name}
              </button>
              {isMarked && (
                <button
                  type="button"
                  onClick={() => onRemove(name)}
                  className="text-[11px] text-text-muted hover:text-red-400"
                  title="Remove mark"
                >
                  ✕
                </button>
              )}
            </div>
          );
        })}
      </div>
      <label className="mt-3 flex items-center justify-between gap-2 text-[11px] text-text-muted">
        Net height
        <select
          value={state.net_height_m}
          onChange={(e) => onNetHeight(Number(e.target.value))}
          className={cn(fieldCls, 'h-7 py-0 text-[11px]')}
        >
          {NET_HEIGHTS.map((h) => (
            <option key={h.value} value={h.value}>
              {h.label}
            </option>
          ))}
        </select>
      </label>
      <p className={cn('mt-3 text-[11px]', fit ? 'text-text-secondary' : 'text-amber-400')}>
        {fit ? `Fit RMSE ${fit.rmse_m.toFixed(2)} m` : state.fit_error}
      </p>
      <p className={cn('mt-1 text-[11px]', camera ? 'text-text-secondary' : 'text-amber-400')}>
        {camera
          ? `Camera at (${camera.center.map((v) => v.toFixed(1)).join(', ')}) m · reprojection ${(camera.rmse * (frameHeight ?? 1080)).toFixed(1)} px${camera.off_floor ? '' : ' · mark the net tops to check height'}`
          : state.camera_error}
      </p>
    </>
  );
}

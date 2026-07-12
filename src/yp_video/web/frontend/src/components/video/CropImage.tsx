import { type DragEvent, type MouseEvent, type ReactNode } from 'react';
import { cn } from '@/lib/cn';

/** COCO-17 limb pairs. */
const SKELETON: Array<[number, number]> = [
  [5, 7], [7, 9], [6, 8], [8, 10], // arms
  [5, 6], [5, 11], [6, 12], [11, 12], // torso
  [11, 13], [13, 15], [12, 14], [14, 16], // legs
  [0, 5], [0, 6], // head
];
// Wrists + ankles get the big red dots (association anchor / court position).
const EMPHASIS = new Set([9, 10, 15, 16]);
const KP_CONF = 0.3;

interface CropImageProps {
  src: string;
  /** Crop-relative [x, y, conf] per COCO keypoint; low-conf joints render gray. */
  keypoints?: Array<[number, number, number]> | null;
  skeleton: boolean;
  /** Sizing classes for the image (e.g. 'h-28 w-auto'). Keep the natural
   *  aspect ratio — the overlay is stretched over the full image box. */
  className?: string;
  alt?: string;
  title?: string;
  draggable?: boolean;
  onDragStart?: (e: DragEvent<HTMLDivElement>) => void;
  onClick?: (e: MouseEvent<HTMLDivElement>) => void;
  onDoubleClick?: (e: MouseEvent<HTMLDivElement>) => void;
  /** Rendered as data-event-id so page-level marquee selection can hit-test. */
  dataId?: string;
  children?: ReactNode;
}

/** A crop thumbnail with an optional SVG skeleton overlay drawn from data —
 *  the jpg itself stays raw, so toggling costs nothing. */
export function CropImage({ src, keypoints, skeleton, className, alt, title, draggable, onDragStart, onClick, onDoubleClick, dataId, children }: CropImageProps) {
  return (
    <div className="relative inline-block" data-event-id={dataId} draggable={draggable} onDragStart={onDragStart} onClick={onClick} onDoubleClick={onDoubleClick} title={title}>
      <img src={src} alt={alt ?? ''} loading="lazy" draggable={false} className={cn('block', className)} />
      {skeleton && keypoints && keypoints.length >= 17 && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          {SKELETON.map(([a, b], i) => {
            const pa = keypoints[a];
            const pb = keypoints[b];
            if (!pa || !pb) return null;
            const solid = pa[2] >= KP_CONF && pb[2] >= KP_CONF;
            return (
              <line
                key={i}
                x1={pa[0] * 100}
                y1={pa[1] * 100}
                x2={pb[0] * 100}
                y2={pb[1] * 100}
                stroke={solid ? '#fff' : 'rgba(190,190,190,0.65)'}
                strokeWidth={solid ? 1.8 : 1}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
          {keypoints.map(([kx, ky, conf], i) => (
            <circle
              key={i}
              cx={kx * 100}
              cy={ky * 100}
              r={conf < KP_CONF ? 1.4 : EMPHASIS.has(i) ? 2.8 : 1.6}
              fill={conf < KP_CONF ? 'rgba(190,190,190,0.8)' : EMPHASIS.has(i) ? '#ff3c3c' : '#e8e83c'}
            />
          ))}
        </svg>
      )}
      {children}
    </div>
  );
}

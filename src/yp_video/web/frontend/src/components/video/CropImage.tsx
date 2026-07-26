import { type DragEvent, type MouseEvent, type ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface CropImageProps {
  src: string;
  /** Sizing classes for the image (e.g. 'h-28 w-auto'). Keep the natural
   *  aspect ratio — the overlay is stretched over the full image box. */
  className?: string;
  alt?: string;
  title?: string;
  draggable?: boolean;
  onDragStart?: (e: DragEvent<HTMLDivElement>) => void;
  onClick?: (e: MouseEvent<HTMLDivElement>) => void;
  onDoubleClick?: (e: MouseEvent<HTMLDivElement>) => void;
  onMouseEnter?: (e: MouseEvent<HTMLDivElement>) => void;
  onMouseLeave?: (e: MouseEvent<HTMLDivElement>) => void;
  /** Rendered as data-event-id so page-level marquee selection can hit-test. */
  dataId?: string;
  children?: ReactNode;
}

/** A crop thumbnail with slots for status overlays. */
export function CropImage({ src, className, alt, title, draggable, onDragStart, onClick, onDoubleClick, onMouseEnter, onMouseLeave, dataId, children }: CropImageProps) {
  return (
    <div className="relative inline-block" data-event-id={dataId} draggable={draggable} onDragStart={onDragStart} onClick={onClick} onDoubleClick={onDoubleClick} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} title={title}>
      <img src={src} alt={alt ?? ''} loading="lazy" draggable={false} className={cn('block', className)} />
      {children}
    </div>
  );
}

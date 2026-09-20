import { useRef, useState } from 'react';
import type { PointerEvent } from 'react';

export type Box = [number, number, number, number];
type Gesture = {
  start: [number, number];
  original: Box[];
  index: number;
  corner: number | null;
  draw: boolean;
};
const clamp = (n: number) => Math.max(0, Math.min(1, n));

/** Normalized SVG coordinates preserve boxes across responsive image sizes. */
export function BoxEditor({
  src,
  boxes,
  onChange,
  disabled,
  selected,
  onSelect,
  draw,
  onReady,
}: {
  src: string;
  boxes: Box[];
  onChange: (boxes: Box[]) => void;
  disabled: boolean;
  selected: number | null;
  onSelect: (index: number | null) => void;
  draw: boolean;
  onReady: (ready: boolean) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const gesture = useRef<Gesture | null>(null);
  const [preview, setPreview] = useState<Box[] | null>(null);
  const [error, setError] = useState(false);
  const point = (e: PointerEvent): [number, number] => {
    const rect = svg.current!.getBoundingClientRect();
    return [
      clamp((e.clientX - rect.left) / rect.width),
      clamp((e.clientY - rect.top) / rect.height),
    ];
  };
  const begin = (e: PointerEvent, index: number | null, corner: number | null = null) => {
    if (disabled || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const start = point(e);
    const drawing = draw || index === null;
    const i = drawing ? boxes.length : index;
    onSelect(i);
    gesture.current = {
      start,
      original: boxes.map((b) => [...b]),
      index: i,
      corner,
      draw: drawing,
    };
    setPreview(drawing ? [...boxes, [...start, ...start]] : boxes);
    svg.current!.setPointerCapture(e.pointerId);
  };
  const move = (e: PointerEvent) => {
    const g = gesture.current;
    if (!g) return;
    const [x, y] = point(e);
    const next = g.original.map((b) => [...b] as Box);
    if (g.draw)
      next.push([
        Math.min(g.start[0], x),
        Math.min(g.start[1], y),
        Math.max(g.start[0], x),
        Math.max(g.start[1], y),
      ]);
    else if (g.corner !== null) {
      const b = next[g.index]!;
      const ax = g.corner % 2 === 0 ? b[2] : b[0];
      const ay = g.corner < 2 ? b[3] : b[1];
      next[g.index] = [Math.min(ax, x), Math.min(ay, y), Math.max(ax, x), Math.max(ay, y)];
    } else {
      const b = next[g.index]!;
      const dx = Math.max(-b[0], Math.min(1 - b[2], x - g.start[0]));
      const dy = Math.max(-b[1], Math.min(1 - b[3], y - g.start[1]));
      next[g.index] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy];
    }
    setPreview(next);
  };
  const finish = () => {
    if (gesture.current && preview) {
      const edited = preview[gesture.current.index];
      const valid = edited && edited[2] - edited[0] > 0.002 && edited[3] - edited[1] > 0.002 ? preview : boxes;
      if (JSON.stringify(valid) !== JSON.stringify(boxes)) onChange(valid);
      if (selected !== null && selected >= valid.length) onSelect(null);
    }
    gesture.current = null;
    setPreview(null);
  };
  return (
    <div className="relative select-none overflow-hidden rounded-lg bg-black">
      <img
        src={src}
        alt="待標註人物的影格"
        draggable={false}
        className="block w-full"
        onLoad={() => {
          setError(false);
          onReady(true);
        }}
        onError={() => {
          setError(true);
          onReady(false);
        }}
      />
      {error ? (
        <p role="alert" className="p-6 text-red-400">
          影格載入失敗，請重新載入。
        </p>
      ) : (
        <svg
          ref={svg}
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
          aria-label="人物框編輯區"
          className={`absolute inset-0 h-full w-full touch-none ${draw ? 'cursor-crosshair' : 'cursor-default'}`}
          onPointerDown={(e) => begin(e, null)}
          onPointerMove={move}
          onPointerUp={finish}
          onPointerCancel={() => {
            gesture.current = null;
            setPreview(null);
          }}
        >
          {(preview ?? boxes).map((b, i) => (
            <g key={i}>
              <rect
                x={b[0] * 1000}
                y={b[1] * 1000}
                width={(b[2] - b[0]) * 1000}
                height={(b[3] - b[1]) * 1000}
                fill={i === selected ? '#38bdf822' : '#10b98111'}
                stroke={i === selected ? '#38bdf8' : '#34d399'}
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
                onPointerDown={(e) => begin(e, i)}
                style={{ cursor: draw ? 'crosshair' : 'move' }}
              />
              <text
                x={b[0] * 1000 + 3}
                y={b[1] * 1000 + 22}
                fill="white"
                stroke="black"
                strokeWidth={0.8}
                fontSize={20}
                pointerEvents="none"
              >
                {i + 1}
              </text>
              {i === selected &&
                !draw &&
                [0, 1, 2, 3].map((c) => (
                  <rect
                    key={c}
                    x={b[c % 2 === 0 ? 0 : 2] * 1000 - 5}
                    y={b[c < 2 ? 1 : 3] * 1000 - 7}
                    width={10}
                    height={14}
                    fill="#38bdf8"
                    stroke="white"
                    vectorEffect="non-scaling-stroke"
                    onPointerDown={(e) => begin(e, i, c)}
                    style={{ cursor: 'nwse-resize' }}
                  />
                ))}
            </g>
          ))}
        </svg>
      )}
    </div>
  );
}

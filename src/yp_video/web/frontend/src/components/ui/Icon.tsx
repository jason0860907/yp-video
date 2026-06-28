import { cn } from '@/lib/cn';

interface IconProps {
  /** One or more Heroicons-outline `d` path strings. */
  paths: string[];
  className?: string;
}

/** Inline line icon — monochrome, inherits currentColor, ~1.8px stroke. */
export function Icon({ paths, className }: IconProps) {
  return (
    <svg
      className={cn('h-[18px] w-[18px] flex-shrink-0', className)}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      {paths.map((d) => (
        <path key={d} strokeLinecap="round" strokeLinejoin="round" d={d} />
      ))}
    </svg>
  );
}

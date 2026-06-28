export type ProgressVariant = 'primary' | 'accent' | 'success';

const GRADIENTS: Record<ProgressVariant, string> = {
  primary: 'linear-gradient(90deg, rgb(var(--primary-dark)), rgb(var(--primary)))',
  accent: 'linear-gradient(90deg, rgb(var(--accent-dark)), rgb(var(--accent)))',
  success: 'linear-gradient(90deg, #2D9A52, #34C759)',
};

const GLOWS: Record<ProgressVariant, string> = {
  primary: '0 0 8px rgb(var(--primary) / 0.45)',
  accent: '0 0 8px rgb(var(--accent) / 0.45)',
  success: '0 0 8px rgba(52,199,89,0.40)',
};

interface ProgressBarProps {
  /** 0..1 */
  progress: number | null | undefined;
  variant?: ProgressVariant;
}

export function ProgressBar({ progress, variant = 'primary' }: ProgressBarProps) {
  const pct = Math.round(Math.max(0, Math.min(progress ?? 0, 1)) * 100);
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
      <div
        className="relative h-full rounded-full transition-all duration-500 ease-out"
        style={{
          width: `${pct}%`,
          background: GRADIENTS[variant],
          boxShadow: pct > 3 ? GLOWS[variant] : undefined,
        }}
      >
        {pct > 5 && <div className="shimmer-bg absolute inset-0 rounded-full" />}
      </div>
    </div>
  );
}

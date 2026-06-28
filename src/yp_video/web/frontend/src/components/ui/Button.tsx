import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/cn';

export type ButtonIntent = 'primary' | 'default' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

const INTENTS: Record<ButtonIntent, string> = {
  primary: 'bg-primary text-white border border-transparent hover:brightness-110',
  default:
    'bg-surface-300 text-text-secondary border border-border-light hover:text-text-primary hover:border-border-bright',
  ghost: 'bg-transparent text-text-secondary border border-transparent hover:text-text-primary hover:bg-white/5',
  danger: 'bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25',
};

const SIZES: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-[11.5px]',
  md: 'px-3.5 py-2 text-[13px]',
  lg: 'px-[18px] py-[11px] text-sm',
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  intent?: ButtonIntent;
  size?: ButtonSize;
  icon?: ReactNode;
}

/** Action button. `primary` is the brand-green CTA; `default` is the dense
 *  toolbar button; `danger` for destructive actions. */
export function Button({
  intent = 'default',
  size = 'md',
  icon,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-lg font-semibold leading-none whitespace-nowrap',
        'cursor-pointer transition-[filter,background-color,border-color,color] duration-150',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary',
        'disabled:opacity-45 disabled:cursor-not-allowed disabled:hover:brightness-100',
        INTENTS[intent],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}

/** Searchable single-select over {value, label} options — VideoCombobox's
 *  interaction model (focus opens, typing narrows, arrows + Enter pick,
 *  Escape closes, ✕ clears) generalized away from video names, for long
 *  option lists like checkpoints. '' means nothing picked.
 */

import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/train/Field';

export interface SearchSelectOption {
  value: string;
  label: string;
}

export function SearchSelect({
  options,
  value,
  onChange,
  placeholder = 'Type to search…',
  className,
}: {
  options: SearchSelectOption[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const needle = query.trim().toLowerCase();
  // Exact > prefix > substring on the label, so a pasted full name always
  // sits at the top. Sort is stable, so list order survives within ranks.
  const rank = (label: string) => (label === needle ? 0 : label.startsWith(needle) ? 1 : 2);
  const shown = needle
    ? options
        .filter((o) => o.label.toLowerCase().includes(needle) || o.value.toLowerCase().includes(needle))
        .sort((a, b) => rank(a.label.toLowerCase()) - rank(b.label.toLowerCase()))
    : options;

  const selected = options.find((o) => o.value === value);

  useEffect(() => setHighlight(0), [needle, open]);
  useEffect(() => {
    listRef.current?.children[highlight]?.scrollIntoView({ block: 'nearest' });
  }, [highlight]);

  const close = () => {
    setOpen(false);
    setQuery('');
  };
  const pick = (next: string) => {
    onChange(next);
    close();
    inputRef.current?.blur();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (!open) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlight((h) => Math.min(h + 1, shown.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlight((h) => Math.max(h - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (shown[highlight]) pick(shown[highlight].value);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      close();
      inputRef.current?.blur();
    }
  };

  return (
    <div className={cn('relative', className)}>
      <input
        ref={inputRef}
        value={open ? query : (selected?.label ?? value)}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setOpen(true)}
        onBlur={close}
        onKeyDown={onKeyDown}
        placeholder={(!open && (selected?.label ?? value)) || placeholder}
        title={value}
        className={cn(fieldCls, 'pr-8 text-ellipsis')}
      />
      {value && !open && (
        <button
          type="button"
          onClick={() => onChange('')}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-text-muted hover:text-text-primary"
        >
          ✕
        </button>
      )}
      {open && (
        <div
          ref={listRef}
          className="absolute left-0 right-0 top-full z-50 mt-1 max-h-72 overflow-auto rounded-xl border border-border bg-surface-100 p-1 shadow-2xl"
        >
          {shown.length === 0 ? (
            <div className="px-3 py-2 text-xs text-text-muted">No match</div>
          ) : (
            shown.map((option, i) => (
              <button
                key={option.value || '∅'}
                type="button"
                // Mousedown (not click) so the pick lands before the input's blur.
                onMouseDown={(e) => {
                  e.preventDefault();
                  pick(option.value);
                }}
                onMouseEnter={() => setHighlight(i)}
                className={cn(
                  'w-full rounded-lg px-3 py-2 text-left text-xs text-text-secondary',
                  i === highlight && 'bg-primary/10 text-text-primary',
                )}
              >
                <span className="block min-w-0 break-all">{option.label}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

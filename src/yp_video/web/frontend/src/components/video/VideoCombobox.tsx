import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface VideoComboboxProps<T extends { name: string }> {
  /** Candidates, already narrowed by any page-level filters (kind, status). */
  items: T[];
  /** Selected name; '' means nothing picked. */
  value: string;
  onChange: (name: string) => void;
  placeholder?: string;
  /** Row content in the dropdown; defaults to the bare name. */
  renderItem?: (item: T) => ReactNode;
  className?: string;
}

const inputCls =
  'h-9 w-full rounded-lg border border-border-light bg-surface-50 px-3 pr-8 font-mono text-xs text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';

/** Searchable single-select picker for video/file names — replaces native
 *  <select> where the option list is long. Focus opens the full list, typing
 *  narrows it, arrows + Enter pick, Escape closes. */
export function VideoCombobox<T extends { name: string }>({
  items,
  value,
  onChange,
  placeholder = 'Type to search…',
  renderItem,
  className,
}: VideoComboboxProps<T>) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const needle = query.trim().toLowerCase();
  const shown = needle ? items.filter((i) => i.name.toLowerCase().includes(needle)) : items;

  useEffect(() => setHighlight(0), [needle, open]);
  useEffect(() => {
    listRef.current?.children[highlight]?.scrollIntoView({ block: 'nearest' });
  }, [highlight]);

  const close = () => {
    setOpen(false);
    setQuery('');
  };
  const pick = (name: string) => {
    onChange(name);
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
      if (shown[highlight]) pick(shown[highlight].name);
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
        value={open ? query : value}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setOpen(true)}
        onBlur={close}
        onKeyDown={onKeyDown}
        placeholder={value || placeholder}
        className={inputCls}
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
          className="absolute left-0 right-0 top-full z-50 mt-1 max-h-96 overflow-auto rounded-xl border border-border bg-surface-100 p-1 shadow-2xl"
        >
          {shown.length === 0 ? (
            <div className="px-3 py-2 text-xs text-text-muted">No videos match</div>
          ) : (
            shown.map((item, i) => (
              <button
                key={item.name}
                type="button"
                // Mousedown (not click) so the pick lands before the input's blur.
                onMouseDown={(e) => {
                  e.preventDefault();
                  pick(item.name);
                }}
                onMouseEnter={() => setHighlight(i)}
                className={cn(
                  'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs text-text-secondary',
                  i === highlight && 'bg-primary/10 text-text-primary',
                )}
              >
                {renderItem ? renderItem(item) : <span className="min-w-0 flex-1 break-all font-mono">{item.name}</span>}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

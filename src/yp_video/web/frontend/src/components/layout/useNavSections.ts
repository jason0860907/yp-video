import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'sidebarOpenSections';

function read(): string[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
    return Array.isArray(stored) ? stored.filter((v): v is string => typeof v === 'string') : [];
  } catch {
    return [];
  }
}

/**
 * Which collapsible sidebar sections are expanded, remembered across reloads.
 *
 * Landing on a route inside a folded section unfolds it — a fold must never
 * hide the page you are looking at. It is an unfold, not a pin: the user can
 * close it again while still on that page.
 */
export function useNavSections(activeSection: string | undefined) {
  const [open, setOpen] = useState<Set<string>>(() => new Set(read()));

  useEffect(() => {
    if (!activeSection) return;
    setOpen((previous) => {
      if (previous.has(activeSection)) return previous;
      return new Set(previous).add(activeSection);
    });
  }, [activeSection]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...open]));
  }, [open]);

  const toggle = useCallback((title: string) => {
    setOpen((previous) => {
      const next = new Set(previous);
      if (!next.delete(title)) next.add(title);
      return next;
    });
  }, []);

  return { isOpen: (title: string) => open.has(title), toggle };
}

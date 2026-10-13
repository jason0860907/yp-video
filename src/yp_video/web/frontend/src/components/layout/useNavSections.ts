import { useCallback, useEffect, useState } from 'react';

/**
 * Which collapsible sidebar sections are expanded — an accordion keyed to the
 * route.
 *
 * Landing on a route unfolds its section and folds every other one, so the
 * list never grows past the window and pushes System under the footer. The
 * user can still open or close any section by hand until the next navigation.
 */
export function useNavSections(activeSection: string | undefined) {
  const [open, setOpen] = useState<Set<string>>(() => new Set(activeSection ? [activeSection] : []));

  useEffect(() => {
    setOpen(new Set(activeSection ? [activeSection] : []));
  }, [activeSection]);

  const toggle = useCallback((title: string) => {
    setOpen((previous) => {
      const next = new Set(previous);
      if (!next.delete(title)) next.add(title);
      return next;
    });
  }, []);

  return { isOpen: (title: string) => open.has(title), toggle };
}

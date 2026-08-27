/** Sidebar list scrolling shared by the Action Label and ReID rally lists.
 *
 *  Both render the same shape — rally rows that expand into their actions —
 *  and both have to follow a selection the user did not make by scrolling:
 *  arriving from another tab, Prev/Next, a jump from the timeline. Keeping one
 *  implementation keeps the two sidebars feeling like one thing.
 */

/** Pin a rally row (or the outside header) to the top of the list.
 *
 *  Rows are found by their `data-rally-row` attribute, and the measurement is
 *  deferred a frame: the same click usually expands or collapses a rally, and
 *  measuring before that re-layout lands scrolls to the old position. */
export function scrollRallyTop(list: HTMLElement | null, key: number | string): void {
  requestAnimationFrame(() => {
    const row = list?.querySelector<HTMLElement>(`[data-rally-row="${CSS.escape(String(key))}"]`);
    if (!list || !row) return;
    list.scrollTo({
      top: row.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop,
      behavior: 'smooth',
    });
  });
}

/** Scroll an action row into the list's view, centering it, but only when
 *  it's actually off-screen — avoids constant re-centering jitter as the
 *  playhead crosses each action during playback. Rows are found by their
 *  `data-action-id` attribute. */
export function scrollActionIntoView(list: HTMLElement | null, id: string): void {
  requestAnimationFrame(() => {
    const row = list?.querySelector<HTMLElement>(`[data-action-id="${CSS.escape(id)}"]`);
    if (!list || !row) return;
    const lr = list.getBoundingClientRect();
    const rr = row.getBoundingClientRect();
    if (rr.top < lr.top || rr.bottom > lr.bottom) {
      list.scrollTo({
        top: rr.top - lr.top + list.scrollTop - lr.height / 2 + rr.height / 2,
        behavior: 'smooth',
      });
    }
  });
}

import { useSyncExternalStore } from 'react';

/** Theme (light/dark) + brand palette, persisted and reflected onto
 *  <html data-theme data-palette>. The initial attributes are also set by an
 *  inline script in index.html to avoid a flash before this module loads. */

export type Theme = 'light' | 'dark';
export type Palette = 'court' | 'sunset' | 'coral';

export const PALETTES: Array<{ key: Palette; label: string; swatch: string }> = [
  { key: 'court', label: 'Court green', swatch: '#2D5F3F' },
  { key: 'sunset', label: 'Sunset orange', swatch: '#E8622B' },
  { key: 'coral', label: 'Coral', swatch: '#E25563' },
];

const THEME_KEY = 'vq-theme';
const PALETTE_KEY = 'vq-palette';

const isTheme = (v: string | null): v is Theme => v === 'light' || v === 'dark';
const isPalette = (v: string | null): v is Palette => v === 'court' || v === 'sunset' || v === 'coral';

let state: { theme: Theme; palette: Palette } = {
  theme: isTheme(localStorage.getItem(THEME_KEY)) ? (localStorage.getItem(THEME_KEY) as Theme) : 'dark',
  palette: isPalette(localStorage.getItem(PALETTE_KEY)) ? (localStorage.getItem(PALETTE_KEY) as Palette) : 'court',
};

const listeners = new Set<() => void>();

function apply() {
  const el = document.documentElement;
  el.setAttribute('data-theme', state.theme);
  el.setAttribute('data-palette', state.palette);
}
apply();

function set(next: Partial<typeof state>) {
  state = { ...state, ...next };
  localStorage.setItem(THEME_KEY, state.theme);
  localStorage.setItem(PALETTE_KEY, state.palette);
  apply();
  listeners.forEach((l) => l());
}

export function toggleTheme() {
  set({ theme: state.theme === 'dark' ? 'light' : 'dark' });
}
export function setPalette(palette: Palette) {
  set({ palette });
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function useTheme() {
  return useSyncExternalStore(
    subscribe,
    () => state,
    () => state,
  );
}

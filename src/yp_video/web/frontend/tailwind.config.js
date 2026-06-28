/** @type {import('tailwindcss').Config} */
// VolleyIQ design system — Pipeline workspace (dark, instrument-grade).
// Brand green #2D5F3F · accent gold #E8B23A · iOS-native dark surfaces.
// Borrowed Tailwind palettes (emerald/red/amber/sky/violet) are remapped to
// the iOS semantic hues so semantic utility classes stay on-brand.
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Brand — palette-switchable via CSS vars (channel triplets → alpha works).
        primary: {
          DEFAULT: 'rgb(var(--primary) / <alpha-value>)',
          light: 'rgb(var(--primary-light) / <alpha-value>)',
          dark: 'rgb(var(--primary-dark) / <alpha-value>)',
          dim: 'rgb(var(--primary) / 0.15)',
        },
        accent: {
          DEFAULT: 'rgb(var(--accent) / <alpha-value>)',
          light: 'rgb(var(--accent-light) / <alpha-value>)',
          dark: 'rgb(var(--accent-dark) / <alpha-value>)',
          dim: 'rgb(var(--accent) / 0.15)',
        },
        // Surfaces / text / lines — theme-switchable (full values, no alpha modifier).
        surface: {
          DEFAULT: 'var(--bg)',
          50: 'var(--surface-50)',
          100: 'var(--surface-100)',
          200: 'var(--surface-200)',
          300: 'var(--surface-300)',
          400: 'var(--surface-400)',
        },
        sidebar: 'var(--sidebar)',
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-muted': 'var(--text-muted)',
        border: { DEFAULT: 'var(--line)', light: 'var(--line-strong)', bright: 'var(--line-bright)' },
        emerald: { 200: '#A7E8C0', 300: '#5FD98C', 400: '#34C759', 500: '#34C759', 600: '#2D9A52' },
        red: { 300: '#FF8A80', 400: '#FF6B5E', 500: '#FF453A', 600: '#E0352B' },
        amber: { 100: '#F3D275', 200: '#F0C868', 300: '#ECBE4F', 400: '#E8B23A', 500: '#E8B23A' },
        sky: { 300: '#7DD3FC', 400: '#38BDF8', 500: '#38BDF8' },
        violet: { 300: '#A78BFA', 400: '#7B5FD4', 500: '#7B5FD4' },
        rose: { 400: '#FF6B5E', 500: '#FF453A' },
        // Six fixed action hues — one per action, identical everywhere.
        action: {
          serve: '#38BDF8', receive: '#22C55E', set: '#A78BFA',
          spike: '#F97316', block: '#EF4444', score: '#FBBF24',
        },
      },
      fontFamily: {
        heading: ['Archivo', 'system-ui', 'sans-serif'],
        body: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        'fade-in': { from: { opacity: '0', transform: 'translateY(6px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        'pulse-dot': { '0%, 100%': { opacity: '1' }, '50%': { opacity: '0.4' } },
        shimmer: { from: { backgroundPosition: '-200% 0' }, to: { backgroundPosition: '200% 0' } },
      },
      animation: {
        'fade-in': 'fade-in 0.3s ease-out',
        'pulse-dot': 'pulse-dot 1.5s ease-in-out infinite',
        shimmer: 'shimmer 2s linear infinite',
      },
    },
  },
  plugins: [],
};

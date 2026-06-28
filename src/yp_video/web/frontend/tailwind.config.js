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
        primary: { DEFAULT: '#2D5F3F', light: '#5BBF8A', dark: '#244D33', dim: 'rgba(45,95,63,0.15)' },
        accent: { DEFAULT: '#E8B23A', light: '#F3D275', dark: '#C8901A', dim: 'rgba(232,178,58,0.15)' },
        surface: { DEFAULT: '#000000', 50: '#0E0E10', 100: '#1C1C1E', 200: '#2C2C2E', 300: '#3A3A3C', 400: '#48484A' },
        'text-primary': '#FFFFFF',
        'text-secondary': 'rgba(235,235,245,0.62)',
        'text-muted': 'rgba(235,235,245,0.40)',
        border: { DEFAULT: 'rgba(84,84,88,0.45)', light: 'rgba(84,84,88,0.65)', bright: 'rgba(120,120,128,0.55)' },
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

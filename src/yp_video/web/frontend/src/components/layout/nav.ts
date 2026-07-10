/** Single source of truth for navigation — drives both the sidebar and the
 *  router. Icon paths are Heroicons-outline `d` strings (1.8px stroke). */

export interface NavItem {
  path: string;
  label: string;
  icon: string[];
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

// Categories mirror the VolleyIQ prototype's Pipeline sidebar.
const ICON = {
  download: ['M12 4v12m0 0l-4-4m4 4l4-4M4 20h16'],
  cut: [
    'M14.121 14.121A3 3 0 109.879 9.879m4.242 4.242L9.879 9.879m4.242 4.242l4.243 4.243M9.879 9.879L5.636 5.636m4.243 4.243L5.636 14.121M14.121 9.879l4.243-4.243',
  ],
  detect: [
    'M15 12a3 3 0 11-6 0 3 3 0 016 0z',
    'M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z',
  ],
  annotate: ['M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z'],
  train: ['M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z'],
  predict: ['M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'],
  pin: [
    'M15 10.5a3 3 0 11-6 0 3 3 0 016 0z',
    'M19.5 10.5c0 7.5-7.5 11.25-7.5 11.25S4.5 18 4.5 10.5a7.5 7.5 0 1115 0z',
  ],
  cloud: ['M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z'],
  jobs: [
    'M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z',
  ],
};

export const NAV: NavSection[] = [
  {
    title: 'Video',
    items: [
      { path: '/download', label: 'Download', icon: ICON.download },
      { path: '/cut', label: 'Cut', icon: ICON.cut },
    ],
  },
  {
    title: 'Rally',
    items: [
      { path: '/detect', label: 'Rally Predict', icon: ICON.detect },
      { path: '/annotate', label: 'Rally Label', icon: ICON.annotate },
      { path: '/spot-train', label: 'Rally SPOT Train', icon: ICON.train },
      { path: '/spot-predict', label: 'Rally SPOT Predict', icon: ICON.predict },
    ],
  },
  {
    title: 'Action',
    items: [
      { path: '/action-predict', label: 'Action Predict', icon: ICON.predict },
      { path: '/action-annotate', label: 'Action Label', icon: ICON.pin },
      { path: '/action-train', label: 'Action Train', icon: ICON.train },
    ],
  },
  {
    title: 'System',
    items: [
      { path: '/upload', label: 'Cloud Storage', icon: ICON.cloud },
      { path: '/jobs', label: 'Jobs', icon: ICON.jobs },
    ],
  },
];

/** Flat list of all routed paths, in sidebar order. */
export const NAV_ITEMS: NavItem[] = NAV.flatMap((s) => s.items);

/** Path → its sidebar category, for the top-bar "{category}/{title}" heading. */
export const PATH_SECTION: Record<string, string> = Object.fromEntries(
  NAV.flatMap((s) => s.items.map((it) => [it.path, s.title])),
);

export const DEFAULT_PATH = '/download';

/** Per-route page title shown in the top bar. */
export const PAGE_TITLES: Record<string, string> = {
  '/download': 'Download',
  '/cut': 'Cut into sets',
  '/detect': 'Rally Predict',
  '/annotate': 'Rally Label',
  '/spot-train': 'Rally SPOT Train',
  '/spot-predict': 'Rally SPOT Predict',
  '/action-train': 'Action Train',
  '/action-predict': 'Action Predict',
  '/action-annotate': 'Action Label',
  '/upload': 'Cloud Storage',
  '/jobs': 'Jobs & System',
};

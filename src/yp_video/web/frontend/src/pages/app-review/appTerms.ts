/** The App's words, colours and formats, so the App views read like the App:
 *  zh-Hant values from VolleyIQ's Localizable.xcstrings, colours from
 *  TouchKind / Palette / RosterPlayer. Review's own chrome stays English. */

import type { Player } from './types';

export const KINDS = ['serve', 'receive', 'set', 'spike', 'block'] as const;
type Kind = (typeof KINDS)[number] | 'score';

/** TouchKind: timeline glyph, spelled-out name, the App's softened hue. */
export const KIND: Record<Kind, { short: string; long: string; color: string }> = {
  serve: { short: '發', long: '發球', color: '#71BADA' },
  receive: { short: '接', long: '接球', color: '#54AD75' },
  set: { short: '舉', long: '舉球', color: '#AF9FDC' },
  spike: { short: '扣', long: '扣球', color: '#DB925F' },
  block: { short: '攔', long: '攔網', color: '#D27575' },
  score: { short: '★', long: '得分', color: '#DDBC68' },
};
export const kindOf = (kind: string) => KIND[kind as Kind] ?? { short: '·', long: '其他', color: '#8E8E93' };

/** Palette: primary (dark mode), accent, danger. */
export const APP = { primary: '#4E9A6A', accent: '#E8B23A', danger: '#FF3B30' };

export const FRAMING: Record<string, string> = {
  full_play: '組織進攻',
  to_next: '單一動作',
  whole_rally: '整個 Rally',
};

/** CourtSide.title — also the teams once the user has set 我方. */
export const SIDE: Record<string, string> = {
  left: '左',
  right: '右',
  near: '近',
  far: '遠',
  us: '我方',
  them: '對手',
};
export const WINNER: Record<string, string> = {
  left: '左側得分',
  right: '右側得分',
  near: '近側得分',
  far: '遠側得分',
};
export const OPPOSITE: Record<string, string> = { left: 'right', right: 'left', near: 'far', far: 'near' };

const LOSS_REASON: Record<string, string> = {
  spike_error: '扣球失誤',
  blocked: '被攔網',
  set_error: '舉球失誤',
  receive_error: '接發失誤',
  chance_ball_error: '嗆司接噴',
  defense_error: '防守失誤',
  serve_error: '發球失誤',
  other: '其他',
};
/** LossReason.displayName: built-ins by code, custom ones carry the user's
 *  own text after the prefix. */
export const lossReasonLabel = (code: string) =>
  LOSS_REASON[code] ?? (code.startsWith('custom:') ? code.slice(7) : code);

/** rallyTimeLabel: m:ss, truncated. */
export function rallyTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** hmsLabel: m:ss, h:mm:ss from an hour. */
export function hms(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 3600) return rallyTime(s);
  return `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

/** RosterPlayer.displayName: the name, or the jersey while unnamed. */
export const displayName = (p: Player) => p.name || `#${p.number}`;

/** SwiftUI's Color(hue:saturation:brightness:) as CSS. */
function hsb(hue: number, s: number, b: number): string {
  const l = b * (1 - s / 2);
  const sl = l === 0 || l === 1 ? 0 : (b - l) / Math.min(l, 1 - l);
  return `hsl(${hue} ${Math.round(sl * 100)}% ${Math.round(l * 100)}%)`;
}
/** RosterPlayer.color — the dot on a player chip. */
export const playerColor = (p: Player) => hsb(p.hue, 0.45, 0.55);
/** RosterPlayer.softColor — behind the player chip on cards. */
export const playerSoftColor = (p: Player) => hsb(p.hue, 0.4, 0.92);
/** Tag.color — a favourite category's chip. */
export const tagColor = (hue: number) => hsb(hue, 0.55, 0.65);

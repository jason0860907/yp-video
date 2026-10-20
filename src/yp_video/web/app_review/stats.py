"""The App's 分析 → 摘要 numbers for one match: a port of ``MatchStats`` /
``MatchFindings`` over the projection's clips, without the parts that need a
court calibration (球路, 球速與高度 and the court findings).

Inputs are what the App reads: every surviving touch of the five kinds
(outside rallies too), one score clip per point, and each rally's winner —
as a team once the user has said where theirs stood (``team_sides``), else
the camera-frame side.
"""

from collections import Counter

from .models import Rally, TeamSides

KINDS = ["serve", "receive", "set", "spike", "block"]
OPPOSITE = {"left": "right", "right": "left", "near": "far", "far": "near", "us": "them", "them": "us"}
# ``LossReason`` group order on the card: teams first, then sides.
SIDE_ORDER = ["us", "them", "left", "right", "near", "far"]
TITLE = {"us": "我方", "them": "對手", "left": "左", "right": "右", "near": "近", "far": "遠"}
MINIMUM_SAMPLE = 4


def team(sides: TeamSides | None, side: str | None, rally_index: int) -> str | None:
    """``TeamSides.team(on:inRally:)``: us / them, nil off the start's axis."""
    if not sides or not side:
        return None
    switched = sum(1 for s in sides.switches if s <= rally_index) % 2
    ours = OPPOSITE[sides.start] if switched else sides.start
    return "us" if side == ours else "them" if side == OPPOSITE[ours] else None


def half_up(x: float) -> int:
    """Foundation's rounding for the loss-reason shares (not banker's)."""
    return int(x + 0.5)


def summary(
    rallies: list[Rally], preview: dict, sides: TeamSides | None, winner
) -> dict:
    ordered = sorted(rallies, key=lambda r: (r.start, r.index))

    def point_side(rally: Rally) -> str | None:
        side = winner(rally)
        return team(sides, side, rally.index) or side

    # Touches and the deciding plays, per rostered player.
    players: dict[int, dict] = {}

    def line(player: dict) -> dict:
        return players.setdefault(
            player["number"],
            {"player": player, "touches": dict.fromkeys(KINDS, 0), "deciding": 0},
        )

    for clip in preview["actions"]:
        if clip["kind"] in KINDS and clip["player"]:
            line(clip["player"])["touches"][clip["kind"]] += 1

    rally_by_index = {r.index: r for r in rallies}
    points = []
    for clip in preview["scores"]:
        if clip["player"]:
            line(clip["player"])["deciding"] += 1
        rally = rally_by_index.get(clip["rally_index"])
        won = winner(rally) if rally else None
        lost = OPPOSITE[won] if won else None
        points.append(
            {
                "losing": (team(sides, lost, rally.index) if rally else None) or lost,
                "losing_team": team(sides, lost, rally.index) if rally else None,
                "reason": clip["loss_reason"],
            }
        )
    has_teams = any(p["losing_team"] for p in points)

    ranked = sorted(
        players.values(), key=lambda p: (-sum(p["touches"].values()), p["player"]["number"])
    )

    # 得分走勢: both sides of every axis a point went to, ordered by title.
    winners = [point_side(r) for r in ordered]
    running = {}
    for w in filter(None, winners):
        running[w] = running[OPPOSITE[w]] = 0
    order = sorted(running, key=lambda s: TITLE[s])
    progression = []
    for position, w in enumerate(winners, 1):
        if w:
            running[w] += 1
        progression.extend({"rally": position, "side": s, "points": running[s]} for s in order)

    # 失分原因: points with both a loser and a reason, grouped by who lost.
    losses = []
    for side in SIDE_ORDER:
        reasons = Counter(p["reason"] for p in points if p["losing"] == side and p["reason"])
        total = sum(reasons.values())
        if total:
            losses.append(
                {
                    "side": side,
                    "total": total,
                    "reasons": [
                        {"reason": code, "count": n, "percent": half_up(n / total * 100)}
                        for code, n in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
                    ],
                }
            )

    # 重點: only 失分最多 reads no court data.
    findings = []
    lost = [p for p in points if p["reason"] and (not has_teams or p["losing_team"] == "us")]
    if lost:
        counts = Counter(p["reason"] for p in lost)
        code, top = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if top >= 2:
            findings.append(
                {"kind": "top_loss_reason", "team": "us" if has_teams else None,
                 "reason": code, "count": top, "total": len(lost)}
            )

    team_score = None
    if sides:
        won = Counter(team(sides, winner(r), r.index) for r in rallies)
        team_score = {"us": won["us"], "them": won["them"]}

    return {
        "rally_count": len(rallies),
        "in_rally_seconds": sum(max(0.0, r.end - r.start) for r in rallies),
        "team_sides": sides.model_dump() if sides else None,
        "team_score": team_score,
        "has_teams": has_teams,
        "findings": findings,
        "progression": progression,
        "players": ranked,
        "losses": losses,
    }

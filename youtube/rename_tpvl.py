#!/usr/bin/env python3
"""
TPVL video batch rename script.

Renames TPVL video files from long titles to concise format.

Original: 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G96 5/17 18:30 台中連莊 vs 桃園雲豹飛將.mp4
Target:   2025-05-17_G96_台中連莊_vs_桃園雲豹飛將.mp4

Usage:
    python -m youtube.rename_tpvl --dry-run    # Preview changes
    python -m youtube.rename_tpvl              # Execute with confirmation
"""

import argparse
import re
import sys
from pathlib import Path

# Default video directory
DEFAULT_VIDEO_DIR = Path.home() / "videos"

# Pattern to match already-formatted files
FORMATTED_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}_G\d+_.+\.mp4$')

# Pattern to match TPVL files
TPVL_PATTERN = re.compile(r'TPVL\s+(\d{4})-(\d{2})')


def parse_tpvl_filename(filename: str) -> dict | None:
    """
    Parse a TPVL video filename to extract metadata.

    Args:
        filename: The original filename

    Returns:
        Dict with game_number, year, month, day, team1, team2, or None if not parseable
    """
    # Skip if already formatted
    if FORMATTED_PATTERN.match(filename):
        return None

    # Must contain TPVL
    if 'TPVL' not in filename:
        return None

    # Extract season (e.g., 2025-26)
    season_match = TPVL_PATTERN.search(filename)
    if not season_match:
        return None
    season_start = int(season_match.group(1))
    season_end = int(season_match.group(2))

    # Extract game number (e.g., G96)
    game_match = re.search(r'G(\d+)', filename)
    if not game_match:
        return None
    game_number = int(game_match.group(1))

    # Extract date (month/day)
    # Handle both regular slash (/) and big solidus (⧸) Unicode character
    date_match = re.search(r'(\d{1,2})[/⧸](\d{1,2})', filename)
    if not date_match:
        return None
    month = int(date_match.group(1))
    day = int(date_match.group(2))

    # Determine year from season
    # 2025-26 season: Aug-Dec is 2025, Jan-Jul is 2026
    if month >= 8:
        year = season_start
    else:
        year = 2000 + season_end

    # Extract teams (after time HH:MM or HH_MM)
    # Pattern: time followed by teams separated by vs
    # Handle both colon (:) and underscore (_) as time separator
    teams_match = re.search(r'\d{1,2}[_:]\d{2}\s+(.+?)(?:\.mp4|$)', filename, re.IGNORECASE)
    if not teams_match:
        return None

    teams_str = teams_match.group(1).strip()
    # Split by vs (case insensitive, with possible spaces)
    teams = re.split(r'\s+vs\.?\s+', teams_str, flags=re.IGNORECASE)
    if len(teams) != 2:
        return None

    team1 = teams[0].strip()
    team2 = teams[1].strip()

    # Remove file extension from team2 if present
    team2 = re.sub(r'\.mp4$', '', team2, flags=re.IGNORECASE)

    # Remove trailing hashtags and text (e.g., "#TPVL")
    team2 = re.sub(r'\s*#.*$', '', team2).strip()

    return {
        'game_number': game_number,
        'year': year,
        'month': month,
        'day': day,
        'team1': team1,
        'team2': team2,
    }


def generate_new_filename(metadata: dict) -> str:
    """
    Generate the new filename from parsed metadata.

    Args:
        metadata: Dict with game_number, year, month, day, team1, team2

    Returns:
        New filename in format: 2025-05-17_G96_台中連莊_vs_桃園雲豹飛將.mp4
    """
    date_str = f"{metadata['year']}-{metadata['month']:02d}-{metadata['day']:02d}"
    game_str = f"G{metadata['game_number']}"
    return f"{date_str}_{game_str}_{metadata['team1']}_vs_{metadata['team2']}.mp4"


def get_unique_filename(directory: Path, filename: str, existing_files: set) -> str:
    """
    Get a unique filename, adding sequence number if needed.

    Args:
        directory: Target directory
        filename: Desired filename
        existing_files: Set of filenames that will exist after rename

    Returns:
        Unique filename (may have _2, _3, etc. suffix)
    """
    if filename not in existing_files and not (directory / filename).exists():
        return filename

    base = filename.rsplit('.mp4', 1)[0]
    counter = 2
    while True:
        new_filename = f"{base}_{counter}.mp4"
        if new_filename not in existing_files and not (directory / new_filename).exists():
            return new_filename
        counter += 1


def scan_and_plan_renames(video_dir: Path) -> list[tuple[Path, str]]:
    """
    Scan directory and plan renames.

    Args:
        video_dir: Directory to scan

    Returns:
        List of (old_path, new_filename) tuples
    """
    renames = []
    new_filenames = set()

    # Get all mp4 files
    mp4_files = sorted(video_dir.glob('*.mp4'))

    for filepath in mp4_files:
        metadata = parse_tpvl_filename(filepath.name)
        if metadata is None:
            continue

        new_filename = generate_new_filename(metadata)
        new_filename = get_unique_filename(video_dir, new_filename, new_filenames)
        new_filenames.add(new_filename)

        if filepath.name != new_filename:
            renames.append((filepath, new_filename))

    return renames


def display_renames(renames: list[tuple[Path, str]]) -> None:
    """Display planned renames."""
    if not renames:
        print("No files to rename.")
        return

    print(f"Found {len(renames)} file(s) to rename:\n")
    for old_path, new_filename in renames:
        print(f"  {old_path.name}")
        print(f"  → {new_filename}\n")


def execute_renames(renames: list[tuple[Path, str]]) -> int:
    """
    Execute the planned renames.

    Args:
        renames: List of (old_path, new_filename) tuples

    Returns:
        Number of files renamed
    """
    count = 0
    for old_path, new_filename in renames:
        new_path = old_path.parent / new_filename
        try:
            old_path.rename(new_path)
            print(f"Renamed: {new_filename}")
            count += 1
        except OSError as e:
            print(f"Error renaming {old_path.name}: {e}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Rename TPVL video files to concise format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run           # Preview changes without renaming
  %(prog)s                     # Rename with confirmation
  %(prog)s --yes               # Rename without confirmation
  %(prog)s -d ~/my-videos      # Use custom directory
        """
    )

    parser.add_argument(
        "-d", "--directory",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Video directory (default: {DEFAULT_VIDEO_DIR})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without renaming"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    video_dir = args.directory.expanduser()
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        sys.exit(1)

    print(f"Scanning: {video_dir}\n")

    renames = scan_and_plan_renames(video_dir)
    display_renames(renames)

    if not renames:
        return

    if args.dry_run:
        print("(Dry run - no files were renamed)")
        return

    if not args.yes:
        try:
            response = input("Proceed with rename? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)

        if response.lower() not in ('y', 'yes'):
            print("Cancelled.")
            return

    print()
    count = execute_renames(renames)
    print(f"\nRenamed {count} file(s).")


if __name__ == "__main__":
    main()

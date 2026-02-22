#!/usr/bin/env python3
"""
YouTube video downloader with audio.

Usage:
    python -m youtube.download <URL> [OPTIONS]
    python -m youtube.download "https://www.youtube.com/watch?v=xxx"
    python -m youtube.download "https://www.youtube.com/watch?v=xxx" --quality 720
    python -m youtube.download "https://www.youtube.com/watch?v=xxx" --audio-only
    python -m youtube.download "https://www.youtube.com/watch?v=xxx" --output ~/videos
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Default output directory
DEFAULT_OUTPUT_DIR = Path.home() / "videos"


def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_ytdlp():
    """Install yt-dlp using pip."""
    print("Installing yt-dlp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"])


def download_video(
    url: str,
    output_dir: str | Path | None = None,
    quality: str = "best",
    audio_only: bool = False,
    format_id: str | None = None,
    filename_template: str = "%(title)s.%(ext)s"
) -> str | None:
    """
    Download YouTube video with audio.

    Args:
        url: YouTube URL
        output_dir: Output directory (default: ~/videos)
        quality: Video quality (best, 1080, 720, 480, 360)
        audio_only: Download audio only
        format_id: Specific format ID (use --list-formats to see options)
        filename_template: Output filename template

    Returns:
        Path to downloaded file, or None if failed
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(output_dir / filename_template)

    cmd = ["yt-dlp", "--cookies", str(Path.home() / "cookies.txt"), "--js-runtimes", "node"]

    if audio_only:
        # Best audio only, convert to mp3
        cmd.extend([
            "-f", "bestaudio",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
        ])
    elif format_id:
        # Use specific format
        cmd.extend(["-f", format_id])
    else:
        # Video + audio merged
        # Use H.264 (avc1) + AAC for maximum compatibility (Mac/Linux/Windows/VSCode)
        if quality == "best":
            cmd.extend([
                "-f", "bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
            ])
        else:
            cmd.extend([
                "-f", f"bestvideo[height<={quality}][vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}]",
                "--merge-output-format", "mp4",
            ])

        # Add faststart for streaming compatibility
        cmd.extend([
            "--ppa", "ffmpeg:-movflags +faststart",
        ])

    cmd.extend([
        "-o", output_template,
        "--no-playlist",  # Download single video, not playlist
        "--progress",
        url
    ])

    print(f"Downloading: {url}")
    print(f"Output: {output_dir}")
    print(f"Quality: {'audio only' if audio_only else quality}")
    print("-" * 50)

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nDownload complete!")
        return str(output_dir)
    else:
        print(f"\nDownload failed with code {result.returncode}")
        return None


def download_youtube_video(url: str, output_dir: str | Path | None = None, quality: str = "best") -> str | None:
    """Convenience function for pipeline use.

    Args:
        url: YouTube URL
        output_dir: Output directory (default: ~/videos)
        quality: Video quality (best, 1080, 720, 480, 360)

    Returns:
        Path to output directory, or None if failed
    """
    # Ensure yt-dlp is available
    if not check_ytdlp():
        install_ytdlp()
        if not check_ytdlp():
            print("Error: Failed to install yt-dlp")
            return None

    return download_video(url=url, output_dir=output_dir, quality=quality)


def list_formats(url: str):
    """List available formats for a video."""
    subprocess.run(["yt-dlp", "-F", url])


def get_video_info(url: str):
    """Get video information."""
    subprocess.run([
        "yt-dlp",
        "--print", "title",
        "--print", "duration_string",
        "--print", "upload_date",
        "--print", "channel",
        url
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos with audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=xxx"              # Best quality
  %(prog)s "https://youtube.com/watch?v=xxx" -q 720       # 720p
  %(prog)s "https://youtube.com/watch?v=xxx" --audio-only # Audio only (MP3)
  %(prog)s "https://youtube.com/watch?v=xxx" -o ~/videos  # Custom output dir
  %(prog)s "https://youtube.com/watch?v=xxx" --list       # List formats
        """
    )

    parser.add_argument(
        "url",
        help="YouTube URL to download"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-q", "--quality",
        default="best",
        choices=["best", "1080", "720", "480", "360"],
        help="Video quality (default: best)"
    )
    parser.add_argument(
        "--audio-only", "-a",
        action="store_true",
        help="Download audio only (MP3)"
    )
    parser.add_argument(
        "-f", "--format",
        dest="format_id",
        help="Specific format ID (use --list to see options)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available formats"
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show video information"
    )
    parser.add_argument(
        "--filename",
        default="%(title)s.%(ext)s",
        help="Output filename template (default: %%(title)s.%%(ext)s)"
    )

    args = parser.parse_args()

    # Check/install yt-dlp
    if not check_ytdlp():
        install_ytdlp()
        if not check_ytdlp():
            print("Error: Failed to install yt-dlp")
            sys.exit(1)

    # Handle different modes
    if args.list:
        list_formats(args.url)
    elif args.info:
        get_video_info(args.url)
    else:
        result = download_video(
            url=args.url,
            output_dir=args.output,
            quality=args.quality,
            audio_only=args.audio_only,
            format_id=args.format_id,
            filename_template=args.filename
        )
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()

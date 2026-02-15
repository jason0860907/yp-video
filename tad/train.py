"""Train TAD model for volleyball rally detection using OpenTAD.

This script wraps OpenTAD training with volleyball-specific configuration.
"""

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


def get_opentad_path() -> Path:
    """Get path to OpenTAD repository."""
    return Path(__file__).parent.parent / "OpenTAD"


def check_opentad_installed() -> bool:
    """Check if OpenTAD is cloned."""
    opentad_path = get_opentad_path()
    return (opentad_path / "tools" / "train.py").exists()


def main():
    parser = argparse.ArgumentParser(description="Train TAD model for volleyball")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "volleyball_actionformer.py",
        help="Config file path",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if not check_opentad_installed():
        print("Error: OpenTAD not found!")
        print("Please clone OpenTAD first:")
        print("  cd yp-video && git clone https://github.com/sming256/OpenTAD.git")
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    opentad_path = get_opentad_path()
    train_script = opentad_path / "tools" / "train.py"

    # Build command
    cmd = [
        sys.executable,
        str(train_script),
        str(args.config.absolute()),
        "--seed",
        str(args.seed),
    ]

    if not args.work_dir:
        today = date.today().strftime("%Y-%m%d")
        args.work_dir = Path(__file__).parent / "checkpoints" / "actionformer" / today

    cmd.extend(["--cfg-options", f"work_dir={args.work_dir.absolute()}"])

    if args.resume:
        cmd.extend(["--resume", str(args.resume.absolute())])

    # Set environment
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(opentad_path)

    # Set distributed training environment for single GPU
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29500"

    print(f"Running: {' '.join(cmd)}")
    print(f"GPU: {args.gpu}")
    print(f"Config: {args.config}")

    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),  # Run from yp-video root
            env=env,
            check=True,
        )
        print("Training complete!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()

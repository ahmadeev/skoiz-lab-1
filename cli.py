import argparse
import math
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenCV and manual Gaussian filters",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=(
                Path(__file__).resolve().parent
                / "images"
                / "input"
                / "Lenna_small.png"
        ),
        help="path to the source image",
    )
    parser.add_argument(
        "-k",
        "--kernel-size",
        type=int,
        default=5,
        help="positive odd kernel size (default: 5)",
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=1.5,
        help="positive Gaussian sigma (default: 1.5)",
    )

    arguments = parser.parse_args()

    if not arguments.image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {arguments.image_path}")

    if arguments.kernel_size <= 0 or arguments.kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number")

    if not math.isfinite(arguments.sigma) or arguments.sigma <= 0:
        raise ValueError("Sigma must be a finite positive number")

    return arguments

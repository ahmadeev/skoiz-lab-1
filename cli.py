import argparse
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "images" / "input" / "Lenna_small.png"


def validate_image_path(image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")


def validate_filter_parameters(kernel_size: int, sigma: float) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number")

    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be a finite positive number")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenCV and manual Gaussian filters",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
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

    validate_image_path(arguments.image_path)
    validate_filter_parameters(arguments.kernel_size, arguments.sigma)

    return arguments

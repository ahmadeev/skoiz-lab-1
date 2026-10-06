import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np

from cli import (
    DEFAULT_IMAGE_PATH,
    PROJECT_ROOT,
    validate_filter_parameters,
    validate_image_path,
)
from main import process_manually, process_with_lib


REPORT_OUTPUT_DIR = PROJECT_ROOT / "helpers" / "output"
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "images" / "output"


@dataclass
class BenchmarkResult:
    kernel_size: int
    sigma: float
    lib_time: float
    manual_time: float
    speedup: float
    max_error: int
    mean_error: float
    equal: bool
    lib_samples: list[float]
    manual_samples: list[float]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenCV and manual Gaussian filters",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="path to the source image",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        type=int,
        default=[3, 5, 7, 15],
        help="kernel sizes for the fixed-sigma experiment",
    )
    parser.add_argument(
        "--fixed-sigma",
        type=float,
        default=1.5,
        help="sigma for the kernel-size experiment",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 3.0],
        help="sigma values for the fixed-kernel experiment",
    )
    parser.add_argument(
        "--fixed-kernel",
        type=int,
        default=7,
        help="kernel size for the sigma experiment",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of measured runs for each filter and parameter set",
    )

    arguments = parser.parse_args()
    validate_arguments(arguments)
    return arguments


def validate_arguments(arguments: argparse.Namespace) -> None:
    validate_image_path(arguments.image_path)

    if arguments.repeats <= 0:
        raise ValueError("Repeats must be positive")

    for kernel_size in arguments.kernels:
        validate_filter_parameters(kernel_size, arguments.fixed_sigma)

    for sigma in arguments.sigmas:
        validate_filter_parameters(arguments.fixed_kernel, sigma)


def measure(function, repeats: int, *args) -> tuple[np.ndarray, list[float]]:
    samples = []
    result = None

    for _ in range(repeats):
        started_at = perf_counter()
        result = function(*args)
        samples.append(perf_counter() - started_at)

    return result, samples


def benchmark_case(
    image: np.ndarray,
    kernel_size: int,
    sigma: float,
    repeats: int,
) -> tuple[BenchmarkResult, np.ndarray, np.ndarray]:
    lib_result, lib_samples = measure(
        process_with_lib,
        repeats,
        kernel_size,
        sigma,
        image,
    )
    manual_result, manual_samples = measure(
        process_manually,
        repeats,
        kernel_size,
        sigma,
        image,
    )

    difference = cv2.absdiff(lib_result, manual_result)
    lib_time = median(lib_samples)
    manual_time = median(manual_samples)

    result = BenchmarkResult(
        kernel_size=kernel_size,
        sigma=sigma,
        lib_time=lib_time,
        manual_time=manual_time,
        speedup=manual_time / lib_time,
        max_error=int(difference.max()),
        mean_error=float(difference.mean()),
        equal=bool(np.array_equal(lib_result, manual_result)),
        lib_samples=lib_samples,
        manual_samples=manual_samples,
    )

    return result, lib_result, manual_result


def format_number(value: float) -> str:
    return f"{value:g}"


def image_name(
    prefix: str,
    kernel_size: int,
    sigma: float,
    source_name: str,
) -> str:
    return (
        f"{prefix}-{kernel_size}x{kernel_size}-"
        f"{format_number(sigma)}__{source_name}"
    )


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    fieldnames = [
        "kernel_size",
        "sigma",
        "lib_median_s",
        "manual_median_s",
        "speedup",
        "max_error",
        "mean_error",
        "equal",
        "lib_samples_s",
        "manual_samples_s",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "kernel_size": result.kernel_size,
                    "sigma": format_number(result.sigma),
                    "lib_median_s": f"{result.lib_time:.6f}",
                    "manual_median_s": f"{result.manual_time:.6f}",
                    "speedup": f"{result.speedup:.2f}",
                    "max_error": result.max_error,
                    "mean_error": f"{result.mean_error:.6f}",
                    "equal": result.equal,
                    "lib_samples_s": ";".join(
                        f"{sample:.6f}" for sample in result.lib_samples
                    ),
                    "manual_samples_s": ";".join(
                        f"{sample:.6f}" for sample in result.manual_samples
                    ),
                }
            )


def markdown_table(results: list[BenchmarkResult]) -> list[str]:
    lines = [
        "| kernel size | sigma | lib, s | manual, s | speedup | max error | mean error | equal |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for result in results:
        lines.append(
            f"| {result.kernel_size} "
            f"| {format_number(result.sigma)} "
            f"| {result.lib_time:.6f} "
            f"| {result.manual_time:.6f} "
            f"| {result.speedup:.2f}x "
            f"| {result.max_error} "
            f"| {result.mean_error:.6f} "
            f"| {str(result.equal).lower()} |"
        )

    return lines


def markdown_image_grid(
    results: list[BenchmarkResult],
    source_name: str,
) -> list[str]:
    lines = []

    for index in range(0, len(results), 2):
        pair = results[index:index + 2]
        images = []
        captions = []

        for result in pair:
            name = image_name(
                "man",
                result.kernel_size,
                result.sigma,
                source_name,
            )
            images.append(f"![result](./images/output/{name})")
            captions.append(
                f"kernel size={result.kernel_size}, "
                f"sigma={format_number(result.sigma)}"
            )

        if len(pair) == 1:
            images.append("")
            captions.append("")

        lines.extend(
            [
                f"| {images[0]} | {images[1]} |",
                "|:---:|:---:|",
                f"| {captions[0]} | {captions[1]} |",
                "",
            ]
        )

    return lines


def write_markdown(
    kernel_results: list[BenchmarkResult],
    sigma_results: list[BenchmarkResult],
    source_name: str,
    fixed_sigma: float,
    fixed_kernel: int,
    repeats: int,
    path: Path,
) -> None:
    lines = [
        f"Измерения выполнены по {repeats} раза. В таблицах указана медиана времени.",
        "",
        f"### Влияние размера ядра при sigma={format_number(fixed_sigma)}",
        "",
        *markdown_table(kernel_results),
        "",
        *markdown_image_grid(kernel_results, source_name),
        f"### Влияние sigma при размере ядра {fixed_kernel}x{fixed_kernel}",
        "",
        *markdown_table(sigma_results),
        "",
        *markdown_image_grid(sigma_results, source_name),
    ]

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    image = cv2.imread(str(arguments.image_path))

    if image is None:
        raise ValueError(f"Failed to decode image: {arguments.image_path}")

    kernel_cases = [
        (kernel_size, arguments.fixed_sigma)
        for kernel_size in arguments.kernels
    ]
    sigma_cases = [
        (arguments.fixed_kernel, sigma)
        for sigma in arguments.sigmas
    ]
    cases = list(dict.fromkeys(kernel_cases + sigma_cases))

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    first_kernel, first_sigma = cases[0]
    process_with_lib(first_kernel, first_sigma, image)

    results = {}

    for index, (kernel_size, sigma) in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] "
            f"kernel_size={kernel_size}, sigma={format_number(sigma)}"
        )

        result, lib_result, manual_result = benchmark_case(
            image,
            kernel_size,
            sigma,
            arguments.repeats,
        )
        results[(kernel_size, sigma)] = result

        lib_name = image_name(
            "lib",
            kernel_size,
            sigma,
            arguments.image_path.name,
        )
        manual_name = image_name(
            "man",
            kernel_size,
            sigma,
            arguments.image_path.name,
        )
        save_image(IMAGE_OUTPUT_DIR / lib_name, lib_result)
        save_image(IMAGE_OUTPUT_DIR / manual_name, manual_result)

        print(
            f"  lib={result.lib_time:.6f} s, "
            f"manual={result.manual_time:.6f} s, "
            f"speedup={result.speedup:.2f}x"
        )

    all_results = [results[case] for case in cases]
    kernel_results = [results[case] for case in kernel_cases]
    sigma_results = [results[case] for case in sigma_cases]
    csv_path = REPORT_OUTPUT_DIR / "benchmark.csv"
    markdown_path = REPORT_OUTPUT_DIR / "benchmark.md"

    write_csv(all_results, csv_path)
    write_markdown(
        kernel_results,
        sigma_results,
        arguments.image_path.name,
        arguments.fixed_sigma,
        arguments.fixed_kernel,
        arguments.repeats,
        markdown_path,
    )

    print(f"CSV report: {csv_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()

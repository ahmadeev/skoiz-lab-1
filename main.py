import cv2
import math
import numpy as np
from functools import wraps
from time import perf_counter

from cli import parse_arguments


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start

        print(f"{func.__name__}: {elapsed:.6f} с")
        return result

    return wrapper


@measure_time
def process_with_lib(size: int, sigma: float, image: np.ndarray):
    return cv2.GaussianBlur(
        src=image,
        ksize=(size, size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_DEFAULT,
    )


@measure_time
def process_manually(size: int, sigma: float, source: np.ndarray):
    result = np.zeros_like(source, dtype=np.float32)

    kernel = create_gaussian_kernel(size, sigma)

    radius = size // 2

    padded = np.pad(
        source,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="reflect",
    )

    height, width = source.shape[:2]

    color = np.zeros(3, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            color.fill(0)

            for dy in range(-radius, radius + 1):
                y_neighbour = y + dy + radius

                for dx in range(-radius, radius + 1):
                    x_neighbour = x + dx + radius

                    weight = kernel[dy + radius, dx + radius]
                    color += padded[y_neighbour, x_neighbour] * weight

            result[y, x] = color

    np.rint(result, out=result)
    np.clip(result, 0, 255, out=result)

    return result.astype(np.uint8)

def create_gaussian_kernel(size: int, sigma: float):
    radius = size // 2
    kernel = np.zeros([size, size], dtype=np.float32)

    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            kernel[y + radius, x + radius] = math.exp(
                -(x * x + y * y) / (2 * sigma * sigma)
            )

    kernel /= kernel.sum()

    return kernel


def main() -> None:
    args = parse_arguments()

    image = cv2.imread(str(args.image_path))

    if image is None:
        raise ValueError(f"Failed to decode image: {args.image_path}")

    lib_result = process_with_lib(args.kernel_size, args.sigma, image)
    cv2.imwrite(f"./images/output/lib-{args.image_path.name}", lib_result)

    man_result = process_manually(args.kernel_size, args.sigma, image)
    cv2.imwrite(f"./images/output/man-{args.image_path.name}", man_result)

    difference = cv2.absdiff(lib_result, man_result)

    print("Максимальная ошибка:", difference.max())
    print("Средняя ошибка:", difference.mean())
    print("Полностью равны:", np.array_equal(lib_result, man_result))

if __name__ == '__main__':
    main()

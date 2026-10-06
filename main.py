import cv2
import math
import numpy as np
from functools import wraps
from time import perf_counter


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
def process_manually(size: int, sigma: float, image: np.ndarray):
    result = np.zeros_like(image, dtype=np.float64)

    kernel = create_gaussian_kernel(size, sigma)

    radius = size // 2

    for y in range(len(image)):
        for x in range(len(image[0])):
            result[y][x] = filter_pixel(x, y, radius, image, kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def filter_pixel(x: int, y: int, radius: int, source: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)

    for dy in range(-radius, radius + 1):
        y_neighbour = y + dy

        if not (0 <= y_neighbour < source.shape[0]):
            y_neighbour = y - dy

        for dx in range(-radius, radius + 1):
            x_neighbour = x - dx if x + dx < 0 else x + dx

            if not (0 <= x_neighbour < source.shape[1]):
                x_neighbour = x - dx

            result += source[y_neighbour, x_neighbour] * kernel[dy + radius][dx + radius]

    return result


def create_gaussian_kernel(size: int, sigma: float):
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    radius = size // 2
    kernel = []

    for y in range(-radius, radius + 1):
        row = []

        for x in range(-radius, radius + 1):
            value = math.exp(
                -(x * x + y * y) / (2 * sigma * sigma)
            )
            row.append(value)

        kernel.append(row)

    total = sum(sum(row) for row in kernel)

    for y in range(size):
        for x in range(size):
            kernel[y][x] /= total

    return kernel


def main() -> None:
    image = cv2.imread("./images/input/demo.png")

    size = 5
    sigma = 1.5

    lib_result = process_with_lib(size, sigma, image)
    cv2.imwrite("./images/output/lib-Lenna.png", lib_result)

    man_result = process_manually(size, sigma, image)
    cv2.imwrite("./images/output/man-Lenna.png", man_result)

    difference = cv2.absdiff(lib_result, man_result)

    print("Максимальная ошибка:", difference.max())
    print("Средняя ошибка:", difference.mean())
    print("Полностью равны:", np.array_equal(lib_result, man_result))

if __name__ == '__main__':
    main()

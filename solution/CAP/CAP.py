import cv2
import numpy as np

from DPC.DCP import recover_scene_radiance


def estimate_depth(
    image,
    theta0=0.121779,
    theta1=0.959710,
    theta2=-0.780245,
):
    """Estimate scene depth with the coefficients learned in the CAP paper."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    depth = theta0 + theta1 * value + theta2 * saturation
    return np.maximum(depth, 0.0).astype(np.float32)


def guided_depth_filter(guide, depth, radius=40, eps=1e-3):
    kernel_size = (2 * radius + 1, 2 * radius + 1)
    border = cv2.BORDER_REFLECT
    mean_guide = cv2.boxFilter(guide, -1, kernel_size, borderType=border)
    mean_depth = cv2.boxFilter(depth, -1, kernel_size, borderType=border)
    mean_product = cv2.boxFilter(
        guide * depth, -1, kernel_size, borderType=border
    )
    mean_guide_squared = cv2.boxFilter(
        guide * guide, -1, kernel_size, borderType=border
    )
    covariance = mean_product - mean_guide * mean_depth
    variance = mean_guide_squared - mean_guide * mean_guide
    coefficient_a = covariance / (variance + eps)
    coefficient_b = mean_depth - coefficient_a * mean_guide
    mean_a = cv2.boxFilter(
        coefficient_a, -1, kernel_size, borderType=border
    )
    mean_b = cv2.boxFilter(
        coefficient_b, -1, kernel_size, borderType=border
    )
    return np.maximum(mean_a * guide + mean_b, 0.0)


def refine_depth(depth, guide, radius=15, guided_radius=40, guided_eps=1e-3):
    if radius < 1 or radius % 2 == 0:
        raise ValueError("radius phải là số nguyên dương và lẻ.")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    local_minimum = cv2.erode(depth, kernel)
    return guided_depth_filter(
        guide, local_minimum, guided_radius, guided_eps
    )


def estimate_atmospheric_light(image, depth, top_percent=0.001):
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent phải nằm trong khoảng (0, 1].")
    flat_depth = depth.ravel()
    flat_image = image.reshape(-1, 3)
    number_of_pixels = max(int(flat_depth.size * top_percent), 1)
    deepest_indices = np.argpartition(
        flat_depth, -number_of_pixels
    )[-number_of_pixels:]
    candidates = flat_image[deepest_indices]
    brightest = np.argmax(np.sum(candidates, axis=1))
    return candidates[brightest].astype(np.float32)


def dehaze(
    image_rgb,
    radius=15,
    beta=1.0,
    t_min=0.1,
    t_max=0.9,
    top_percent=0.001,
    guided_radius=40,
    guided_eps=1e-3,
    theta0=0.121779,
    theta1=0.959710,
    theta2=-0.780245,
):
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có dạng H x W x 3.")
    if beta <= 0:
        raise ValueError("beta phải lớn hơn 0.")
    if not 0 < t_min <= t_max <= 1:
        raise ValueError("Cần 0 < t_min <= t_max <= 1.")

    if image_rgb.dtype == np.uint8:
        image = image_rgb.astype(np.float32) / 255.0
    else:
        image = image_rgb.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        image = np.clip(image, 0.0, 1.0)

    depth_raw = estimate_depth(image, theta0, theta1, theta2)
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    depth = refine_depth(
        depth_raw,
        guide,
        radius,
        guided_radius,
        guided_eps,
    )
    atmospheric_light = estimate_atmospheric_light(
        image, depth, top_percent
    )
    transmission = np.clip(np.exp(-beta * depth), t_min, t_max)
    recovered = recover_scene_radiance(
        image,
        transmission,
        atmospheric_light,
        t0=t_min,
    )
    recovered_uint8 = np.round(recovered * 255.0).astype(np.uint8)
    diagnostics = {
        "atmospheric_light": atmospheric_light,
        "depth_raw": depth_raw,
        "depth": depth,
        "transmission": transmission,
    }
    return recovered_uint8, diagnostics

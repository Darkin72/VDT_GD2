import cv2
import numpy as np


def dark_channel(image, patch_size=15):
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size phải là số nguyên dương và lẻ.")
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    return cv2.erode(min_channel, kernel)


def estimate_atmospheric_light(image, dark, top_percent=0.001):
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent phải nằm trong khoảng (0, 1].")
    flat_image = image.reshape(-1, 3)
    flat_dark = dark.ravel()
    number_of_pixels = max(int(flat_dark.size * top_percent), 1)
    brightest_indices = np.argpartition(
        flat_dark, -number_of_pixels
    )[-number_of_pixels:]
    atmospheric_light = np.mean(flat_image[brightest_indices], axis=0)
    return np.maximum(atmospheric_light, 1e-6).astype(np.float32)


def estimate_transmission(
    image,
    atmospheric_light,
    patch_size=15,
    omega=0.95,
):
    normalized_image = image / atmospheric_light.reshape(1, 1, 3)
    transmission = 1.0 - omega * dark_channel(
        normalized_image, patch_size
    )
    return np.clip(transmission, 0.0, 1.0).astype(np.float32)


def guided_filter(guide, source, radius=40, eps=1e-3):
    if radius < 1:
        raise ValueError("radius phải là số nguyên dương.")
    guide = guide.astype(np.float32, copy=False)
    source = source.astype(np.float32, copy=False)
    kernel_size = (2 * radius + 1, 2 * radius + 1)
    border = cv2.BORDER_REFLECT

    mean_guide = cv2.boxFilter(
        guide, -1, kernel_size, borderType=border
    )
    mean_source = cv2.boxFilter(
        source, -1, kernel_size, borderType=border
    )
    mean_guide_source = cv2.boxFilter(
        guide * source, -1, kernel_size, borderType=border
    )
    mean_guide_squared = cv2.boxFilter(
        guide * guide, -1, kernel_size, borderType=border
    )

    covariance = mean_guide_source - mean_guide * mean_source
    variance = mean_guide_squared - mean_guide * mean_guide
    coefficient_a = covariance / (variance + eps)
    coefficient_b = mean_source - coefficient_a * mean_guide

    mean_a = cv2.boxFilter(
        coefficient_a, -1, kernel_size, borderType=border
    )
    mean_b = cv2.boxFilter(
        coefficient_b, -1, kernel_size, borderType=border
    )
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def recover_scene_radiance(
    image,
    transmission,
    atmospheric_light,
    t0=0.1,
):
    safe_transmission = np.maximum(
        transmission, t0
    )[..., np.newaxis]
    atmospheric_light = atmospheric_light.reshape(1, 1, 3)
    recovered = (
        (image - atmospheric_light) / safe_transmission
        + atmospheric_light
    )
    return np.clip(recovered, 0.0, 1.0)


def dehaze(
    image_rgb,
    patch_size=15,
    omega=0.95,
    t0=0.1,
    top_percent=0.001,
    use_guided_filter=True,
    guided_radius=40,
    guided_eps=1e-3,
):
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có dạng H × W × 3.")

    if image_rgb.dtype == np.uint8:
        image = image_rgb.astype(np.float32) / 255.0
    else:
        image = image_rgb.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        image = np.clip(image, 0.0, 1.0)

    dark = dark_channel(image, patch_size)
    atmospheric_light = estimate_atmospheric_light(
        image, dark, top_percent
    )
    transmission_raw = estimate_transmission(
        image,
        atmospheric_light,
        patch_size,
        omega,
    )

    if use_guided_filter:
        guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        transmission = guided_filter(
            guide,
            transmission_raw,
            guided_radius,
            guided_eps,
        )
    else:
        transmission = transmission_raw

    recovered = recover_scene_radiance(
        image,
        transmission,
        atmospheric_light,
        t0,
    )
    recovered_uint8 = np.round(recovered * 255.0).astype(np.uint8)
    diagnostics = {
        "atmospheric_light": atmospheric_light,
        "dark_channel": dark,
        "transmission_raw": transmission_raw,
        "transmission": transmission,
    }
    return recovered_uint8, diagnostics

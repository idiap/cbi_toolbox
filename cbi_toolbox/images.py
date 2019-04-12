import cv2
import numpy as np


def erase_corners(image_array, corner_size=300):
    if corner_size > 0:
        image_array[..., :corner_size, :corner_size] = image_array[..., corner_size, corner_size, None, None]
        image_array[..., :corner_size, -corner_size:] = image_array[..., corner_size, -corner_size, None, None]
        image_array[..., -corner_size:, -corner_size:] = image_array[..., -corner_size, -corner_size, None, None]
        image_array[..., -corner_size:, :corner_size] = image_array[..., -corner_size, corner_size, None, None]

    return image_array


def transmission_to_absorption(image_array, max_value=4096):
    scale_factor = max_value / np.log(max_value)

    scaled = np.log(max_value / (image_array + 1)) * scale_factor

    return scaled.astype(image_array.dtype)


def remove_background_illumination(image_array, threshold=250, morph_open_size=250, morph_dilate_size=100):
    mask_bool = image_array < threshold
    mask_int = mask_bool.astype(np.uint8)

    kernel_open = np.ones((morph_open_size, morph_open_size), dtype=np.uint8)
    kernel_dilate = np.ones((morph_dilate_size, morph_dilate_size), dtype=np.uint8)

    for plane_idx, plane_mask in enumerate(mask_int):

        _, contours, _ = cv2.findContours(plane_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        plane_mask.fill(1)

        for index, contour in enumerate(contours):
            if len(contour) > 500 and (contour[..., 0].max() == image_array.shape[1] - 1 or contour[..., 0].min() == 0):
                temp_mask = np.ones_like(mask_int[plane_idx, ...])
                cv2.drawContours(temp_mask, contours, index, 0, cv2.FILLED)

                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel_open)
                mask_int[plane_idx, ...] &= temp_mask

        mask_int[plane_idx, ...] = cv2.morphologyEx(mask_int[plane_idx, ...], cv2.MORPH_DILATE, kernel_dilate)

    mask = mask_int.astype(np.uint16)
    mask *= (2 ** 16 - 1)
    image_array &= mask

    return image_array


# def shift_axis_correlation(image, project_axis, padding=100, max_shift=100, rel_corr_threshold=0.5):
#     project = image.sum(project_axis)
#     if padding > 0:
#         project = project[..., padding:-padding]
#
#     project = project - project.mean()
#     p_mean = project.mean(0)
#
#     if max_shift > 0:
#         project = project[..., max_shift:-max_shift]
#
#     shifts = np.empty(project.shape[0], dtype=int)
#     correlations = np.empty(shifts.shape)
#
#     for index, sub_project in enumerate(project):
#         correlation = np.correlate(p_mean, sub_project)
#         shift = correlation.argmax()
#         correlations[index] = correlation[shift]
#         shifts[index] = shift
#     shifts = shifts - max_shift
#     correlations = correlations / correlations.max()
#     shifts[correlations < rel_corr_threshold] = 0
#
#     return shifts

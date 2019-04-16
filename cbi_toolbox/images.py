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

    scaled_array = image_array / max_value

    absorption = np.log(1 / scaled_array)

    return absorption


def remove_background_illumination(image_array, threshold=0.5, hole_size=250, margin_size=100):
    mask_bool = image_array < threshold
    mask_int = mask_bool.astype(np.uint8)

    kernel_open = np.ones((hole_size, hole_size), dtype=np.uint8)
    kernel_dilate = np.ones((margin_size, margin_size), dtype=np.uint8)

    for plane_idx, plane_mask in enumerate(mask_int):
        find_contours = cv2.findContours(plane_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        major = cv2.__version__.split('.')[0]
        if major == '2':
            contours = find_contours[1]
        else:
            contours = find_contours[0]

        plane_mask.fill(0)

        for index, contour in enumerate(contours):
            if len(contour) > 500 and (contour[..., 0].max() == image_array.shape[1] - 1 or contour[..., 0].min() == 0):
                temp_mask = np.zeros_like(mask_int[plane_idx, ...])
                cv2.drawContours(temp_mask, contours, index, 1, cv2.FILLED)

                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel_open)
                mask_int[plane_idx, ...] |= temp_mask

        mask_int[plane_idx, ...] = cv2.morphologyEx(mask_int[plane_idx, ...], cv2.MORPH_ERODE, kernel_dilate)

    mask_bool = mask_int.astype(bool)
    image_array[mask_bool] = 0

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

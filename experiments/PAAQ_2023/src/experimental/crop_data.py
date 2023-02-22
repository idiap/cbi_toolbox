# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>
#
# This file is part of CBI Toolbox.
#
# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the 3-Clause BSD License.
#
# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# 3-Clause BSD License for more details.
#
# You should have received a copy of the 3-Clause BSD License along
# with CBI Toolbox. If not, see https://opensource.org/licenses/BSD-3-Clause.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import numpy as np
import apeer_ometiff_library.io as omeio
import matplotlib.pyplot as plt
import json
import cv2
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)
    parser.add_argument("name")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dwidth", type=int, default=1024)

    args = parser.parse_args()

    data_path = pathlib.Path(args.data)

    if args.output_path is None:
        output_path = data_path / "output"
    else:
        output_path = pathlib.Path(args.output_path)

    output_path = output_path / args.name

    refs = None
    sigs = None

    if args.multimodal:
        channels = ("red", "green", "white")
    else:
        channels = ("red", "green")

    for c, channel in enumerate(channels):
        print(f"Reading channel: {channel}")

        c_array, xmlstring = omeio.read_ometiff(
            str(data_path / channel / "images_MMStack_Default.ome.tif")
        )
        c_array = c_array.squeeze()

        # Dropping first frame due to camera gain...
        c_ref = c_array[2::2]
        c_sig = c_array[3::2]

        if refs is None:
            refs = np.empty_like(c_ref, shape=(len(channels), *c_ref.shape))
            sigs = np.empty_like(c_ref, shape=(len(channels), *c_ref.shape))

        refs[c] = c_ref
        sigs[c] = c_sig

        del c_ref
        del c_sig
        del c_array

    half_w = args.width // 2

    if args.multimodal:
        ref_norm = refs[:-1].mean() / refs[-1].mean()
        refs[-1] = refs[-1] * ref_norm

    x_l = [refs.shape[-1] // 2]
    y_l = [refs.shape[-2] // 2]
    y = y_l[-1]
    x = x_l[-1]

    print("Choosing crop")

    def onclick(event):
        if event.button == 1:
            x = int(event.xdata)
            y = int(event.ydata)
            x_l.append(x)
            y_l.append(y)

        ax[1, 0].cla()
        ax[1, 0].imshow(refs[0, 0, y - half_w : y + half_w, x - half_w : x + half_w])

        for c, _ in enumerate(channels):
            ax[1, c + 1].cla()
            ax[1, c + 1].imshow(
                sigs[c, 0, y - half_w : y + half_w, x - half_w : x + half_w]
            )

        plt.draw()

    fig, ax = plt.subplots(2, len(channels) + 1)
    ax[0, 0].imshow(refs[0, 0, ::, ::])
    ax[0, 0].set_title("Reference")
    ax[1, 0].imshow(refs[0, 0, y - half_w : y + half_w, x - half_w : x + half_w])
    fig.suptitle("Click on top images to center view, exit when ready")

    for c, channel in enumerate(channels):
        ax[0, c + 1].imshow(sigs[c, 0, ::, ::])
        ax[0, c + 1].set_title(f"Channel: {channel}")
        ax[1, c + 1].imshow(
            sigs[c, 0, y - half_w : y + half_w, x - half_w : x + half_w]
        )

    for axis in ax.ravel():
        axis.axis("off")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    y_c = y_l[-1]
    x_c = x_l[-1]
    width = 2 * half_w

    print("Computing affine matrix")

    margin = args.margin
    refs_ = refs[
        :,
        :,
        y_c - half_w - margin : y_c + half_w + margin,
        x_c - half_w - margin : x_c + half_w + margin,
    ].copy()
    sigs_ = sigs[
        :,
        :,
        y_c - half_w - margin : y_c + half_w + margin,
        x_c - half_w - margin : x_c + half_w + margin,
    ].copy()

    mrefs = refs_.mean(1)
    mrefs /= mrefs.max()
    mrefs *= 255
    mrefs = mrefs.astype(np.uint8)

    cv2.setRNGSeed(0)

    orb = cv2.ORB_create()
    kp1, descsA = orb.detectAndCompute(mrefs[0], None)
    kp2, descsB = orb.detectAndCompute(mrefs[1], None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descsA, descsB, None)

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[: int(len(matches) * 0.9)]

    p1 = np.zeros((len(matches), 2))
    p2 = np.zeros((len(matches), 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    trans, mask_t = cv2.estimateAffinePartial2D(p1, p2)

    h, w = refs_.shape[-2:]

    r_aligned = np.empty_like(refs_[0])
    s_aligned = np.empty_like(sigs_[0])

    print("Aligning channels")

    for idx, image in enumerate(refs_[0]):
        r_aligned[idx] = cv2.warpAffine(image, trans, (w, h), flags=cv2.INTER_CUBIC)
        s_aligned[idx] = cv2.warpAffine(
            sigs_[0, idx], trans, (w, h), flags=cv2.INTER_CUBIC
        )

    refs_[0] = r_aligned
    sigs_[0] = s_aligned

    del r_aligned
    del s_aligned
    refs_ = refs_[:, :, margin:-margin, margin:-margin]
    sigs_ = sigs_[:, :, margin:-margin, margin:-margin]

    print(f"Creating output path {str(output_path)}")
    output_path.mkdir(parents=True, exist_ok=True)

    print("Saving cropped data")
    np.savez(str(output_path / "data_cropped"), reference=refs_, signal=sigs_)

    del refs_
    del sigs_

    parameters = {
        "x_c": x_c,
        "y_c": y_c,
        "width": width,
    }

    print("Saving cropping parameters")
    with (output_path / "cropping.json").open("w") as fp:
        json.dump(parameters, fp)

    print("Data cropping done")

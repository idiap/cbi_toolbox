# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib_scalebar import scalebar
from cbi_toolbox.reconstruct import scale_to_mse
from string import ascii_lowercase

matplotlib.use("svg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["pdf.fonttype"] = 42

margin = False

na = 50


path = os.environ["OVC_PATH"]

if margin:
    path = os.path.join(path, "margin")

rpath = os.path.join(path, "reconstruct")
npath = os.path.join(path, "noise")
gpath = os.path.join(path, "graph")


ref = np.load(os.path.join(path, "arrays", "phantom.npy"))

radon = np.load(os.path.join(rpath, "iradon.npy"))

fss = np.load(os.path.join(npath, "fssopt_{:03d}.npy".format(na)))
fps = np.load(os.path.join(npath, "fpsopt_{:03d}.npy".format(na)))
deconv = np.load(os.path.join(npath, "{:03d}_{:03d}.npy".format(na, na)))

scale_to_mse(fps, fss)


arrays = [ref, fss, fps, deconv]


def array_to_image(arr):
    layer = arr.shape[0] // 2
    return arr[layer, ...]


cmap = "magma_r"
cmap = "binary_r"


titles = list(ascii_lowercase)
titles = ["Reference", "FSS-OPT", "FPS-OPT, no deconv", "FPS-OPT, BW deconv"]

figsize = (4, 4.3)

scale = 1
figsize = [d * scale for d in figsize]

fig, plots = plt.subplots(2, 2, figsize=figsize)

images = []
mini = None
maxi = None
for arr in arrays:
    scale_to_mse(ref, arr)
    im = array_to_image(arr)
    images.append(im)
    if mini is None or im.min() < mini:
        mini = im.min()
    if maxi is None or im.max() > maxi:
        maxi = im.max()


mini = 0

for idx, (img, plot, title) in enumerate(zip(images, plots.flatten(), titles)):
    im = plot.imshow(img.T, cmap=cmap, origin="lower", vmin=mini, vmax=maxi)

    width = 0.4
    axin = plot.inset_axes([0.0, 0.0, width, width])
    axin.imshow(
        img.T, interpolation="nearest", cmap=cmap, origin="lower", vmin=mini, vmax=maxi
    )

    origin = 110, 42
    width = 30

    x1, x2, y1, y2 = origin[0], origin[0] + width, origin[1], origin[1] + width

    axin.set_xlim(x1, x2)
    axin.set_ylim(y1, y2)
    axin.set_xticks([])
    axin.set_yticks([])

    plot.indicate_inset_zoom(axin, edgecolor="0.5")

    plot.set_title(title)

    plot.set_xticks([])
    plot.set_yticks([])

    if idx == 0:
        scale = scalebar.ScaleBar(
            0.635, "um", location=2, box_color="k", color="w", box_alpha=0
        )
        plot.add_artist(scale)

plt.subplots_adjust(wspace=0.0, hspace=0.16, left=0, right=1, top=0.945, bottom=0)
plt.savefig(os.path.join(gpath, "images_noise.pdf"))

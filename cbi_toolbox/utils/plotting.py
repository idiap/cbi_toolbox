"""
The plotting module contains helper functions to plot animated movies in matplotlib.
"""

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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class AnimImshow:
    """
    Create an animated figure over an array of images.

    Parameters
    ----------
    images : np.ndarray
        Array containing the images, shape [t, x, y].
    interval : int, optional
        Frame duration in ms, by default 100.
    """

    def __init__(self, images, interval=100):
        self.interval = interval
        self.images = images
        self.fig = plt.figure()
        self.im = plt.imshow(images[0, ...], animated=True)
        self.ani = animation.FuncAnimation(
            self.fig,
            self._updatefig,
            interval=interval,
            frames=images.shape[0],
            blit=True,
        )

    def _updatefig(self, anim_index):
        self.im.set_array(self.images[anim_index, ...])
        return self.im

    def save_to_gif(self, path):
        """
        Save the animated figure as a gif file.

        Parameters
        ----------
        path : str
            Path to the file to save.
        """

        self.ani.save(path, writer="imagemagick", fps=1000 / self.interval)


def trace_points(points, coordinates):
    """
    Compute point trace images from coordinates. Can be then used to plot
    point trajectory.

    Parameters
    ----------
    points : np.ndarray(N, 2)
        Coordinates of the N points for which to draw traces.
    coordinates : np.ndarray(T, 2, W, H)
        Coordinate space in which the points are tracked, given as a sequence
        of T meshgrids.

    Returns
    -------
    np.ndarray(N, T, 2)
        Array containing for each point the position of the closest point
        in the input coordinate arrays, at each step T in the sequence.
    """

    points = np.atleast_2d(points)

    dist = np.power(points[:, None, :, None, None] - coordinates[None, ...], 2).sum(2)

    traces = dist.reshape((dist.shape[0], dist.shape[1], -1)).argmin(-1)
    traces = np.unravel_index(traces, dist.shape[-2:])

    return np.array(traces).transpose((1, 2, 0))

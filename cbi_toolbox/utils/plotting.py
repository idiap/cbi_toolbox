"""
The plotting module contains helper functions to plot animated movies in matplotlib.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>

# This file is part of CBI Toolbox.

# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CBI Toolbox. If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
            self.fig, self._updatefig, interval=interval, frames=images.shape[0], blit=True)

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

        self.ani.save(path, writer='imagemagick', fps=1000/self.interval)

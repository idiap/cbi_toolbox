"""
The plotting module contains helper functions to plot animated movies in matplotlib.
"""

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

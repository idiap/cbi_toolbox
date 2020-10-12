import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AnimImshow:
    def __init__(self, images, interval=100):
        self.interval = interval
        self.images = images
        self.fig = plt.figure()
        self.im = plt.imshow(images[0, ...], animated=True)
        self.ani = animation.FuncAnimation(
            self.fig, self._updatefig, interval=interval, frames=images.shape[0], blit=True)

    def _updatefig(self, anim_index, *args):
        self.im.set_array(self.images[anim_index, ...])
        return self.im,

    def save_to_gif(self, path):
        self.ani.save(path, writer='imagemagick', fps=1000/self.interval)

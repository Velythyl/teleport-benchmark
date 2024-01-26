import io

from matplotlib import pyplot as plt


class Plt2Gif:
    def __init__(self, path=None):
        self.frames = []

        if path is None:
            path = f"/tmp/plt2gif"

        self.basepath = path

    def resolve_path(self, path1, path2):
        if path1 is None:
            assert path2 is not None
            return path2
        return path1

    @property
    def counter(self):
        return len(self.frames)

    def plt_show(self, save, show, path=None):
        import numpy as np

        fig = plt.gcf()
        if save:
            path = self.resolve_path(path, f"{self.basepath}/fig{self.counter}.png")
            fig.savefig(path)
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))

        self.frames.append(im)

        if show:
            plt.show()

    def get_gif(self, save, show, path=None):
        from matplotlib import animation
        fig, ax = plt.subplots()
        ax.axis("off")
        fig.tight_layout()

        im = ax.imshow(self.frames[0], interpolation='none', aspect='auto', vmin=0, vmax=1)

        def animate(i):
            im.set_array(self.frames[i + 1])
            return [im]

        anime = animation.FuncAnimation(fig, animate, frames=self.counter - 1)
        if save:
            path = self.resolve_path(path, f"{self.basepath}/evolution.png")
            anime.save(path, fps=10)
        if show:
            plt.show()

    def reset(self):
        self.frames = []

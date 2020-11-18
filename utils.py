import numpy as np


class DrawLine():

    def __init__(self, env, title, xlabel=None, ylabel=None):
        import visdom

        self.vis = visdom.Visdom()
        self.update_flag = False
        self.env = env
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def __call__(
            self,
            xdata,
            ydata,
    ):
        if not self.update_flag:
            self.win = self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                opts=dict(
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    title=self.title,
                ),
                env=self.env,
            )
            self.update_flag = True
        else:
            self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                win=self.win,
                env=self.env,
                update='append',
            )



class Display:
    def __init__(self):
        import pygame
        self.pygame = pygame

        self.pygame.init()
        self.size = (400, 400)
        self.display_surface = self.pygame.display.set_mode(self.size)

    @staticmethod
    def gray(im):
        im = 255 * (im / im.max())
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
        return ret

    def show(self, arr, scale=True):
        white = (255, 255, 255)
        assert arr.ndim in [2, 3], "incorrect dimensions"

        arr = arr.T

        # unnormalize
        if arr.dtype in [np.float, np.float32, np.float64]:
            arr -= arr.min()
            arr /= arr.max()
            arr *= 255
            arr = arr.astype(np.int)

        # gray to color
        if arr.ndim == 2:
            arr = self.gray(arr)

        surf = self.pygame.surfarray.make_surface(arr)

        if scale:
            surf = self.pygame.transform.scale(surf, self.size)

        self.display_surface.fill(white)
        self.display_surface.blit(surf, (0, 0))
        self.pygame.display.update()

    def close(self):
        self.pygame.quit()

import matplotlib as mpl
try:
    mpl.use('Qt5Agg')
except ValueError as e:
    print('Error: matplotlib backend\n', e)
    print('Trying:', mpl.get_backend())
    mpl.use(matplotlib.get_backend())

import argparse
import array
import os
import time
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from numba import autojit

neumann_n = [(-1,0),(1,0),(0,1),(0,-1),(0,0)]

moore_n = [(-1,0),(1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,1),(0,0)]

Neighborhood =  { "moore" : moore_n,
                 "neumann": neumann_n
}

@autojit
def model(neighbors):
    if np.sum(neighbors) > 5 or np.sum(neighbors) == 4:
        return 1
    else:
        return 0

class Grid(object):
    def __init__(self, gridsize, n_steps, rule, neighborhood, n_state = 2, init_state='random'):
        self.Z = np.zeros((gridsize, gridsize),"uint8")
        if init_state == 'random':
            self.Z = np.random.randint(n_state, size=(gridsize, gridsize))
        else:
            self.Z[gridsize // 2, gridsize // 2] = 1
        self.dim = gridsize
        self.steps = n_steps
        self.rule = rule
        self.n_state = n_state
        self.neighborhood = Neighborhood[neighborhood]
    
    def set_initial(self):
        self.Z = np.zeros((self.dim, self.dim),"uint8")
        self.Z[self.dim // 2, self.dim // 2] = 1
    
    @autojit
    def update(self):
        newZ = np.zeros((self.dim, self.dim),"uint8")
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                neighbors = []
                for dx,dy in self.neighborhood:
                    x = i + dx
                    y = j + dy
                    if x >= self.dim:
                        x = 0
                    if y >= self.dim:
                        y = 0
                    if x < 0:
                        x = self.dim-1
                    if y < 0:
                        y = self.dim-1    
                    neighbors.append(self.Z[x,y])
                newZ[i,j] = model(neighbors)
        self.Z = newZ
        return self.Z

    def init_plot(self):
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 8, True)

        cmap = cm.nipy_spectral
        cmap.set_over((1., 0., 0.))
        cmap.set_under((0., 0., 1.))
        bounds = list(x for x in range(0, self.n_state))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        self.im = self.ax.imshow(self.Z, cmap=cmap,animated=True, clim=(0, self.n_state))
        # self.time0 = time.time()
        return self.fig, self.ax
        
    @autojit
    def update_plot(self,i):
        self.im.set_array(self.update())
        # if i == self.steps-1:
            # print(("t= %s seconds " % (time.time() - self.time0)))
        return self.im

n_steps = 20
G = Grid(100, n_steps, 999, 'moore', 2, 'random')
fig, ax = G.init_plot()
anim = animation.FuncAnimation(fig,
                                   G.update_plot,
                                   np.arange(1, n_steps),
                                   repeat=False,
                                   blit=False,
                                   interval=100)
plt.axis('off')
plt.tight_layout()

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=5, bitrate=1800)
    # anim.save('moore.mp4', writer=writer)
plt.show()
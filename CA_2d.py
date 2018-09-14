#import matplotlib
#matplotlib.use('Agg')
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
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import numba

#first item cell iteself
neumann_n = [(0,0),(0,1),(0,-1),(-1,0),(1,0)]

moore_n = [(0,0),(-1,0),(1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

Neighborhood =  { "moore" : moore_n,
                 "neumann": neumann_n
}

n_steps = 1600
gridsize = 100
n_state = 3
neighborhood_type = 'neumann'
neighborhood = Neighborhood[neighborhood_type]


#Neural Activity functions
@numba.jit(nopython=True)
def R(x):
    if x == 1:
        return 2
    else:
        return 0
    
@numba.jit(nopython=True)
def D(x, l):
    k = 0
    if 1 in l:
        k = 1
    else:
        k = 0
    if x == 0:
        return k
    else:
        return 0
    
@numba.jit(nopython=True)
def model(neighbors, cell_1):
    cell = neighbors[0]
    return ((R(cell) + D(cell, neighbors[1:]) - cell_1))%3

def set_initial():
    Z = np.zeros((gridsize, gridsize),"uint8")
    Z[40,40] = 1
    Z[49,46] = 1
    Z[41,53] = 1
    Z[45,45] = 2
    
def update_plot(i):
    im.set_array(update())
    return im

@numba.jit(nopython=True)
def evolve():
    Z = np.zeros((gridsize, gridsize),"uint8")
    t = 0

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8, True)

    cmap = cm.Wistia
    cmap.set_over((1., 0., 0.))
    cmap.set_under((0., 0., 1.))
    bounds = list(x for x in range(0, n_state))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(Z, cmap=cmap,animated=True, clim=(0, n_state))
    newZ = Z.copy()
    for i in range(0, gridsize):
        for j in range(0, gridsize):
            neighbors = []
            cell_1 =  history[t-1][i,j]
            for (dx,dy)in neighborhood:
                x = i + dx
                y = j + dy
                if x >= gridsize:
                    x = 0
                if y >= gridsize:
                    y = 0
                if x < 0:
                    x = gridsize-1
                if y < 0:
                    y = gridsize-1    
                neighbors.append(Z[x,y])
            newZ[i,j] = model(neighbors, cell_1)
        
    Z = newZ.copy()
    t = t + 1
    history[t] = Z.copy()
    return Z

if __name__ == "__main__":
    set_initial()
    history = {-1:np.zeros((gridsize, gridsize),"uint8"), 0:Z.copy()}
    anim = animation.FuncAnimation(fig,
                                   update_plot,
                                   np.arange(1, n_steps),
                                   repeat=False,
                                   blit=False,
                                   interval=100)
    plt.axis('off')
    plt.tight_layout()

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=150, bitrate=1800)
    #anim.save('moore.mp4', writer=writer)
    plt.show()
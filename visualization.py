import numpy as np
import time
import itertools
from pylab import *
from l2distance import l2distance
from numpy.linalg import multi_dot
from numpy import linalg as LA
import numpy.random as random
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def show_pads(pad1, pad2, N,patch_color=None):

    pad1 = pad1.reshape(N, N)
    pad2 = pad2.reshape(N, N)

    fig = plt.figure(figsize=(2,2),dpi=150)
    if patch_color:
        fig.patch.set_facecolor(patch_color)
        fig.patch.set_alpha(0.8)
    plt.grid(True)
    #ax1 = fig.add_subplot(2, 2, 1, aspect='equal')
    ax1 = plt.subplot(221, aspect='equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.pcolor(pad1, vmin=-1, vmax=1,edgecolors='k', linewidths=1,cmap="Blues")
    plt.gca().invert_yaxis()
    
    #ax2 = fig.add_subplot(2, 2, 2, aspect='equal')
    ax2 = plt.subplot(222, aspect='equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.pcolor(pad2, vmin=-1, vmax=1,edgecolors='k', linewidths=1,cmap="Blues")
    plt.gca().invert_yaxis() 


def plot_magnexation_extrema(unique_magnezation, ratio):
    fig, axes = plt.subplots(figsize=(6,6),dpi=128)
    axes.set_xticks(np.arange(len(unique_magnezation)))
    axes.set_yticks(np.arange(len(unique_magnezation)))
    axes.set_xticklabels(unique_magnezation)
    axes.set_yticklabels(unique_magnezation)
    im=axes.pcolormesh(ratio)
    cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return

def plot_extremas(extremas):
    """
    INPUT:
    extremas, a M1 by M2 array, with extremas[i,j] is the number of extremas found for ith pad1 with jth pad2
    """
    plt.figure(figsize=(5, 5),dpi=128)
    plt.pcolor(extremas, vmin=0, vmax=np.max(extremas))
    plt.axes().set_aspect('equal')
    plt.show()

def show_pads_neighbor(pad1, pad2, neighbors):
    k_neighbor = len(neighbors)
    tot = 2 + k_neighbor
    if len(pad1.shape) != 2:
        pad1 = pad1.reshape(N, N)
        pad2 = pad2.reshape(N, N)
        

    fig, axes = plt.subplots(nrows=1, ncols=tot,dpi=100)
    plt.grid(True)

    ax1 = plt.subplot(1,tot,1, aspect='equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.title.set_text("pad1")
    plt.pcolor(pad1, vmin=-1, vmax=1,edgecolors='k', linewidths=1)
    plt.gca().invert_yaxis()

    ax2 = plt.subplot(1,tot,2, aspect='equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.title.set_text("pad2")
    plt.pcolor(pad2, vmin=-1, vmax=1,edgecolors='k', linewidths=1)
    plt.gca().invert_yaxis()
    
    bar = np.max(np.abs(neighbors))

    for i in range(k_neighbor):
        ax3 = plt.subplot(1,tot,3 + i, aspect='equal')
        ax3.set_xticks([])
        ax3.set_yticks([])
        im=plt.pcolor(neighbors[i], vmin=-bar, vmax=bar, cmap='coolwarm',edgecolors='k', linewidths=1)
       
        plt.gca().invert_yaxis()
        ax3.title.set_text(str(i) + "-NN")
        
    cbar_ax = plt.gcf().add_axes([0.95, 0.2, 0.02, 0.6])
    plt.colorbar(cax=cbar_ax,shrink=0.5)
    plt.show()
    plt.clf()
    
    return 


def show_outer_product(outer, name=None, show_spin=False):
    fig = plt.figure(figsize=(1,1),dpi=120)
    ax3 = plt.subplot(111, aspect='equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.pcolor(outer, vmin=-1, vmax=1,edgecolors='k', linewidths=1)
    if show_spin:
        for y in range(outer.shape[0]):
            for x in range(outer.shape[1]):
                if outer[y, x] > 0:
                    plt.text(x + 0.5, y + 0.5, int(outer[y, x]),
                 horizontalalignment='center',
                 verticalalignment='center',fontsize=12
                 )
                else:
                    plt.text(x + 0.5, y + 0.5, int(outer[y, x]),
                 horizontalalignment='center',
                 verticalalignment='center', color='w',fontsize=12
                 )
    plt.gca().invert_yaxis()
    
    plt.show()
    if name is not None:
        plt.savefig(name)
    plt.clf()

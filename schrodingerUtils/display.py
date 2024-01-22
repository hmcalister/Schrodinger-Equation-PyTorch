import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotSurface(X, Y, surface, ax: Axes3D = None, savePath: str = None, showPlot: bool = True):
    """
    Given the meshgrid dimensions X and Y, as well as the surface (e.g. a potential) over those meshgrid dimensions, plot it.

    Note, this function cannot operate over other dimension sizes, as we only have three dimensions to plot in!
    """

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, surface)
    if savePath is not None:
        plt.savefig(savePath)
    if showPlot:
        plt.show()

def plotColormesh(X, Y, Z, ax = None, savePath: str = None, showPlot: bool = True):
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

    ax.pcolormesh(X, Y, Z)
    if savePath is not None:
        plt.savefig(savePath)
    if showPlot:
        plt.show()
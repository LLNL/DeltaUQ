import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import torch


contourParams = dict(
    zdir='z',
    alpha=0.5,
    zorder=1,
    antialiased=True,
    cmap=cm.RdBu
)

surfaceParams = dict(
    rstride=1,
    cstride=1,
    linewidth=0.1,
    edgecolors='k',
    alpha=0.5,
    antialiased=True,
    cmap=cm.RdBu
)
contour_settings = dict(contourParams)
surface_settings = dict(surfaceParams)


def plot3d(f, bounds):
    no_of_grid_samples = 100
    x1 = np.linspace(bounds[0],bounds[1], no_of_grid_samples)
    x2 = np.linspace(bounds[0],bounds[1], no_of_grid_samples)
    X1, X2 = np.meshgrid(x1, x2)
    
    X_grid = torch.from_numpy(np.concatenate((X1.ravel().reshape(-1,1), X2.ravel().reshape(-1,1)),1)).type(torch.FloatTensor)
    y_grid = f(X_grid).detach().numpy()
    Zt = y_grid.reshape(no_of_grid_samples,no_of_grid_samples)

    fig = plt.figure(figsize=(4.5,5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.patch.set_alpha(0.0)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    surf = ax.plot_surface(X1, X2, Zt, **surface_settings)
    contour_settings['offset'] = np.min(Zt)
    ax.view_init(elev=10, azim=30)
    plt.axis('off')
    plt.grid(b=None)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-11,-3)
    plt.show()

def plot2d(f, bounds):
    fig,ax = plt.subplots(figsize=(6,4))
    X = torch.linspace(bounds[0],bounds[1],500).view(-1,1)
    ax.plot(X.detach().numpy(),f(X).detach().numpy(),color = 'darkgreen', linestyle = '--',linewidth=0.8)
    ax.grid(linestyle='--',color='lightgray',linewidth=0.5)
    plt.show()


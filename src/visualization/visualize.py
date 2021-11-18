import matplotlib.pyplot as plt
import matplotlib.animation as anim

import numpy as np

import sklearn.manifold

from src.common import find_limits, merge_into_flat_one

def visualize_signal_pairs(t, inputs, results, labels=None, N=None):
    if N is None:
        N = len(results)
    
    plt.figure(figsize=(8,8))
    
    for i, (x, x_hat) in enumerate(zip(inputs[:N], results[:N])):
        plt.subplot(N,1,i+1)
        plt.plot(t.ravel(),x.ravel(), label='$x$')
        plt.plot(t.ravel(),x_hat.ravel(), label='$\hat{x}$')
        plt.xlabel('time (s)', fontsize=20)
        plt.ylabel('amplitude', fontsize=20)
        plt.legend()
    
    plt.suptitle('Input & Outputs', fontsize=26)
    plt.tight_layout()
    plt.show()

def visualize_latent_space(latent_space, features=None):
    # if more than 2 dimension apply t-SNE to reduce
    if latent_space.shape[-1] > 2:
        ls_reduced = sklearn.manifold.TSNE(latent_space)
        xaxis='TSNE - \phi'
    # if 2 dimension then just copy original
    else:
        ls_reduced = latent_space.copy()
        xaxis='\phi'

    # then plot (iterate over features if available)
    if features is not None:
        N = len(features)
        plt.figure(figsize=(8,8))
        for i,f in enumerate(features):
            plt.subplot(N,1,i+1)
            plt.scatter(latent_space[:,0], latent_space[:,1], c=features[f])
            plt.title(f'{f}')
            plt.xlabel(f'${xaxis}_1$')
            plt.ylabel(f'${xaxis}_2$')
            plt.colorbar()
        plt.suptitle('Latent Space Visualization against Features', fontsize=20)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8,8))
        plt.scatter(latent_space[:,0], latent_space[:,1])
        plt.title(f'Latent Space Visualization')
        plt.xlabel(f'${xaxis}_1$')
        plt.ylabel(f'${xaxis}_2$')
        plt.tight_layout()
        plt.show()


def animate_signal_pairs(t, inputs, results, labels=None):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()
    ax.set_xlim(find_limits(t))
    ax.set_ylim(find_limits(merge_into_flat_one(inputs, results)))
    line1, = ax.plot([],[])
    line2, = ax.plot([],[])
    
    def init():
        line1.set_data([],[])
        line2.set_data([],[])
        return line1,line2,
    
    def animate(i):
        line1.set_data(t,inputs[i])
        line2.set_data(t,results[i])
        return line1,line2,
    
    animation = anim.FuncAnimation(fig, animate, frames=len(inputs), init_func=init, interval=10, blit=True)
    plt.close()
    return animation
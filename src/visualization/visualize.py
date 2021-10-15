import matplotlib.pyplot as plt

def visualize_signal_pairs(t, results, labels=None):
    N = len(results)
    
    plt.figure(figsize=(8,8))
    
    for i, (x, x_hat)  in enumerate(results):
        plt.subplot(N,1,i+1)
        plt.plot(t,x, label='$x$')
        plt.plot(t,x_hat, label='$\hat{x}$')
        plt.xlabel('time (s)', fontsize=20)
        plt.ylabel('amplitude', fontsize=20)
        plt.legend()
    
    plt.suptitle('Input & Outputs', fontsize=26)
    plt.tight_layout()
    plt.show()
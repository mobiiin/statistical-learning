import matplotlib.pyplot as plt
import numpy as np


def plot_sigmoid(beta1):
    x = np.linspace(-5,5,100)
    s = 1 / (1 + np.exp(- beta1 * x))
    plt.plot(x, s, color="orange")
    plt.title('beta:%f' %beta1)
    plt.show()


plot_sigmoid(1)
# plot_sigmoid(10)
# plot_sigmoid(100)

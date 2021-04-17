import numpy as np
import matplotlib.pyplot as plt


def ols(X, y, theta):
    """OLS cost function"""
    # Initialisation of useful values
    m = np.size(y)
    # Cost function in vectorized form
    h = np.dot(X, theta)
    return float((1. / (2 * m)) * np.dot((h - y).T, (h - y)))


def cost_function(x, y, q):
    s = np.power(np.abs(x), q) + np.power(np.abs(y), q)
    return s


def dataset():
    # Creating the dataset
    x = np.linspace(0, 1, 40)
    noise = 1 * np.random.uniform(size=40)
    y = np.sin(x * 1.5 * np.pi)
    y = (y + noise).reshape(-1, 1)

    # Subtracting the mean so that the y's are centered
    y = y - y.mean()
    X = np.vstack((2 * x, x ** 2)).T

    # Normalizing the design matrix to obtain better visualization
    X = X / np.linalg.norm(X, axis=0)

    # Setup of meshgrid of theta values
    xx, yy = np.meshgrid(np.linspace(-5, 20, 125), np.linspace(-20, 5, 125))
    return xx, yy, X, y


def z_value(q):
    xx, yy, X, y = dataset()
    # Computing the cost function for each theta combination
    zz_reg_p = np.array([cost_function(xi, yi, q) for xi, yi in zip(np.ravel(xx), np.ravel(yy))])

    zz_ls = np.array([ols(X, y.reshape(-1, 1), np.array([t0, t1]).reshape(-1, 1))
                      for t0, t1 in zip(np.ravel(xx), np.ravel(yy))])  # least square cost function

    # Reshaping the cost values
    Z_reg_p = zz_reg_p.reshape(xx.shape)
    Z_ls = zz_ls.reshape(xx.shape)
    return Z_ls, Z_reg_p


def plot_reg_p(q=2):
    xx, yy, X, y = dataset()
    Z_ls, Z_reg_p = z_value(q)
    # Plotting the contours
    plt.contour(xx, yy, Z_reg_p, levels=[.5, 1.5, 3, 6, 9, 15, 30, 60, 100, 150, 250], cmap='gist_gray')
    plt.contour(xx, yy, Z_ls, levels=[.01, .06, .09, .11, .15], cmap='coolwarm')
    plt.xlabel('theta 1')
    plt.ylabel('theta 2')
    plt.title('regression and regularization contours - OLS and $\\beta$^%f' %q)

    # Plotting the minimum
    min_ls = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    plt.plot(min_ls[0], min_ls[1], marker='x', color='red', markersize=10)
    plt.plot(0, 0, marker='o', color='black', markersize=10)
    plt.show()


for q in [.5, 1, 2, 3, 4]:
    plot_reg_p(q)

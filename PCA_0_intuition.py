
import matplotlib.pyplot as plt
import numpy as np


# generate white data, 2D Normal(0,1)
_x = np.random.randn(2, 1000)
plt.scatter(_x[0,:], _x[1,:])
plt.axis('equal')
plt.show()


def make_scale_matrix(lambdas):
    return np.diag(lambdas)


def make_rotation_matrix(deg):
    rad = deg * np.pi / 180.0
    c, s = np.cos(rad), np.sin(rad)
    return np.array(((c, -s), (s, c)))


# transform the white data using a matrix 
s = make_scale_matrix([2, 1])
r = make_rotation_matrix(45)
m = r @ s # scale first, then rotate?
x = m @ _x
plt.scatter(x[0,:], x[1,:])
plt.axis('equal')
plt.show()


# get the correlation matrix of the data 
cor = x @ x.T / (x.shape[1] - 1)
eigval, eigvec = np.linalg.eig(cor)
plt.scatter(x[0,:], x[1,:])
plt.quiver(*np.array([[0, 0], [0, 0]]), *(eigvec @ np.diag(eigval)), color=['orange','orange'], scale=15)
plt.axis('equal')
plt.show()


# the original data was never translated so we don't have to subtract mean
# we do need to divide by the variance, scale factor
# also the data is assumed to be normal, because i generated it so. 
x2 = x.copy()
x2[0,:] /= cor[0,0]
x2[1,:] /= cor[1,1]
plt.scatter(x2[0,:], x2[1,:])
plt.quiver(*np.array([[0, 0], [0, 0]]), *(eigvec @ np.diag(eigval)), color=['orange','orange'], scale=15)
plt.axis('equal')
plt.show()


# now project the data on the highest eigenvector 
v = eigvec[:, [0]] # its norm is 1 
#x3 = (np.array([[1], [0]]).T @ v) * v
x3 = (x2.T @ v).squeeze() * v
plt.scatter(x3[0,:], x3[1,:])
plt.quiver(*np.array([[0, 0], [0, 0]]), *(eigvec @ np.diag(eigval)), color=['orange','orange'], scale=15)
plt.axis('equal')
plt.show()


# explained variance 
eigval[0] / np.sum(eigval)
# 80% of the variance is explained by the factor with variance 2 (2^2 + 1^2 = 5, 4/5 = .8)

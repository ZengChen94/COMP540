import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################

  for i in xrange(m):
    for k in xrange(theta.shape[1]):
      temp = theta.T.dot(X[i])
      maxVal = temp[np.argmax(temp)]
      P = np.exp(theta[:, k].T.dot(X[i]) - maxVal) / np.sum(np.exp(theta.T.dot(X[i]) - maxVal))
      # P = np.exp(theta[:, k].T.dot(X[i])) / np.sum(np.exp(theta.T.dot(X[i])))
      J -= int(y[i] == k) * np.log(P) / m
      grad[:, k] -= X[i] * (int(y[i] == k) - P) / m

  J += reg * np.sum(theta * theta) / (2 * m)
  grad += reg * theta / m


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################


  matrix = X.dot(theta)
  matrix -= np.max(matrix)
  matrix = np.exp(matrix)
  matrix /= np.sum(matrix, axis=1)[:,None]

  label = np.array(range(theta.shape[1]))
  Y = (np.ones((X.shape[0], theta.shape[1])) * label).T
  L = (Y == y).T

  J = -np.sum(np.log(matrix[L == 1])) / m + reg * np.sum(theta * theta) / (2 * m)
  grad = -X.T.dot(L - matrix) / m + reg * theta / m

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

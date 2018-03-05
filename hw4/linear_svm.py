import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  y_hx = np.multiply(y, X.dot(theta))
  zeros_vec = np.zeros([m, 1])
  J = np.sum(theta ** 2) / (2 * m) + C * np.mean(np.maximum(1 - y_hx, zeros_vec))
  grad = theta / m - C*1.0 / m * (y * (y_hx < 1)).dot(X)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
  for i in range(m):
    scores = np.dot(X[i], theta)
    correct_scores = scores[y[i]]
    for j in range(K):
      if j != y[i]:
        margin = scores[j] - correct_scores + delta
        J += max(0, margin)

  J = J / m + 0.5 * reg * np.sum(theta * theta)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  for i in range(m):
    scores = np.dot(X[i], theta)
    correct_scores = scores[y[i]]
    for j in range(K):
      if j != y[i]:
        margin = scores[j] - correct_scores + delta
        if margin > 0:
          dtheta[:, j] += X[i]
          dtheta[:, y[i]] -= X[i]

  dtheta = dtheta / m + reg * theta

  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  K = theta.shape[1]  # number of classes
  m = X.shape[0]  # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################

  scores = X.dot(theta)
  correct_scores = np.array([[scores[i][y[i]]] * K for i in range(m)])
  margin = scores - correct_scores + (scores != correct_scores) * delta
#   sumTmp = np.sum(margin > 0, axis=1)
  sumTmp = np.sum((margin > 0) * margin)
  J = sumTmp / m + 0.5 * reg * np.sum(theta * theta)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  sumTmp = np.sum(margin > 0, axis=1)
  tmp = np.multiply(X.T, sumTmp).T
  dtheta = (margin > 0).T.dot(X).T / m - (margin == 0).T.dot(tmp).T / m + reg * theta

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta

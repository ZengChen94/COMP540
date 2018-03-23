import numpy as np


def affine_forward(x, theta, theta0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (m, d_1, ..., d_k) and contains a minibatch of m
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension d = d_1 * ... * d_k, and
  then transform it to an output vector of dimension h.

  Inputs:
  - x: A numpy array containing input data, of shape (m, d_1, ..., d_k)
  - theta: A numpy array of weights, of shape (d, h)
  - theta0: A numpy array of biases, of shape (h,)
  
  Returns a tuple of:
  - out: output, of shape (m, h)
  - cache: (x, theta, theta0)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # 2 lines of code expected 
#   print x.shape
  x_reshape = x.reshape(x.shape[0], np.prod(x.shape[1:]))
  out = np.dot(x_reshape, theta) + theta0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (m, h)
  - cache: Tuple of:
    - x: Input data, of shape (m, d_1, ... d_k)
    - theta: Weights, of shape (d,h)
    - theta0: biases, of shape (h,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (m, d1, ..., d_k)
  - dtheta: Gradient with respect to theta, of shape (d, h)
  - dtheta0: Gradient with respect to theta0, of shape (1,h)
  """
  x, theta, theta0 = cache
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Hint: do not forget to reshape x into (m,d) form
  # 4-5 lines of code expected
  x_reshape = x.reshape(x.shape[0], np.prod(x.shape[1:])) 
#   calculate according to chain rule and dimension
  dtheta = x_reshape.T.dot(dout)
  dx = dout.dot(theta.T).reshape(x.shape)
  dtheta0 = np.sum(dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # 1 line of code expected
  out = np.multiply(x, x>0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # 1 line of code expected. Hint: use np.where
  out = np.multiply(x, x>0)
  dx = (out>0) * dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # 2 lines of code expected
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    # 1 line of code expected
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    # 1 line of code expected
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, theta, theta0, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of m data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (F, C, HH, WW)
  - theta0: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (m, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, theta, theta0, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  m, C, H, W = x.shape
  F, C, HH, WW = theta.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
    
  H_out = 1 + (H + 2*pad - HH)/stride
  W_out = 1 + (W + 2*pad - WW)/stride
#   out = np.zeros(m, F, H_out, W_out)
  out = np.zeros((m, F, H_out, W_out))

  x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
  H += 2*pad
  W += 2*pad
    
  for i in range(m):
#     x_data.shape = C, H, W
    x_data = x_pad[i]
    x_cnt = -1
    y_cnt = -1
    for j in range(0, H-HH+1, stride):
      y_cnt += 1
      for k in range(0, W-WW+1, stride):
        x_cnt += 1
        x_data_sample = x_data[:, j : j+HH, k : k+WW]
        for l in range(0, F):
          conv_value = np.sum(x_data_sample * theta[l]) + theta0[l]
          out[i, l, y_cnt, x_cnt] = conv_value
      x_cnt = -1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, theta, theta0, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dtheta: Gradient with respect to theta
  - dtheta0: Gradient with respect to theta0
  """
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

# inspired by: https://github.com/martinkersner/cs231n/blob/master/assignment2/layers.py

  x, theta, theta0, conv_param = cache
  m, C, H, W = x.shape
  F, C, HH, WW = theta.shape
  m, F, Hdout, Wdout = dout.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  
  H_out = 1 + (H + 2*pad - HH)/stride
  W_out = 1 + (W + 2*pad - WW)/stride  
  
  x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
  
  dx = np.zeros((x.shape))
#   dtheta = np.zeros((dtheta.shape))
  dtheta = np.zeros((F, C, HH, WW))
  dtheta0 = np.zeros((theta0.shape))
  
  for i in range(F):
    dtheta0[i] = np.sum(dout[:, i, :, :])

  for i in range(F):
    for j in range(C):
      for k in range(HH):
        for l in range(WW):
          dtheta[i, j, k, l] = np.sum(dout[:, i, :, :] * x_pad[:, j, k : k+Hdout*stride : stride, l : l+Wdout*stride : stride])
  
  for m_num in range(m):
    for i in range(H):
      for j in range(W):
        for f in range(F):
          for k in range(Hdout):
            for l in range(Wdout):
              mask1 = np.zeros_like(theta[f, :, :, :])
              mask2 = np.zeros_like(theta[f, :, :, :])
              if (i + pad - k * stride) < HH and (i + pad - k * stride) >= 0:
                mask1[:, i + pad - k * stride, :] = 1.0
              if (j + pad - l * stride) < WW and (j + pad - l * stride) >= 0:
                mask2[:, :, j + pad - l * stride] = 1.0
              theta_masked = np.sum(theta[f, :, :, :] * mask1 * mask2, axis=(1, 2))
              dx[m_num, :, i, j] += dout[m_num, f, k, l] * theta_masked
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (m, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  m, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = 1 + (H - pool_height)/stride 
  W_out = 1 + (W - pool_width)/stride 
  out = np.zeros((m, C, H_out, W_out))
    
  for i in range(0, m):
    x_data = x[i]
    xx, yy = -1, -1
    for j in range(0, H-pool_height+1, stride):
      yy += 1
      for k in range(0, W-pool_width+1, stride):
        xx += 1
        x_data_sample = x_data[:, j : j+pool_height, k : k+pool_width]
        for l in range(0, C):
          out[i, l, yy, xx] = np.max(x_data_sample[l])
      xx = -1

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################

# inspired by: https://github.com/martinkersner/cs231n/blob/master/assignment2/layers.py

  x, pool_param = cache
  m, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
    
  dx = np.zeros((x.shape))
  H_out = 1 + (H - pool_height)/stride
  W_out = 1 + (W - pool_width)/stride

  for i in range(m):
    x_data = x[i]
    xx, yy = -1, -1
    for j in range(0, H-pool_height+1, stride):
      yy += 1
      for k in range(0, W-pool_width+1, stride):
        xx += 1
        x_data_sample = x_data[:, j:j+pool_height, k:k+pool_width]
        for l in range(C):
          x_pool = x_data_sample[l]
          mask = x_pool == np.max(x_pool)
          dx[i, l, j : j+pool_height, k : k+pool_width] += dout[i, l, yy, xx] * mask
      xx = -1
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  m = x.shape[0]
  correct_class_scores = x[np.arange(m), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(m), y] = 0
  loss = np.sum(margins) / m
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(m), y] -= num_pos
  dx /= m
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  m = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(m), y])) / m
  dx = probs.copy()
  dx[np.arange(m), y] -= 1
  dx /= m
  return loss, dx

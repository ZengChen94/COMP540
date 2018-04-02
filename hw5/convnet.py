import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


# class ConvNet(object):
#     """
#     A convolutional network with the following architecture:
#
#     conv - relu - 2x2 max pool - affine - relu - affine - softmax
#
#     The network operates on minibatches of data that have shape (N, C, H, W)
#     consisting of N images, each with height H and width W and with C input
#     channels.
#     """
#
#     def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=5,
#                  hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
#                  dtype=np.float32):
#         """
#         Initialize a new network.
#
#         Inputs:
#         - input_dim: Tuple (C, H, W) giving size of input data
#         - num_filters: Number of filters to use in the convolutional layer
#         - filter_size: Size of filters to use in the convolutional layer
#         - hidden_dim: Number of units to use in the fully-connected hidden layer
#         - num_classes: Number of scores to produce from the final affine layer.
#         - weight_scale: Scalar giving standard deviation for random initialization
#           of weights.
#         - reg: Scalar giving L2 regularization strength
#         - dtype: numpy datatype to use for computation.
#         """
#         self.params = {}
#         self.reg = reg
#         self.dtype = dtype
#
#         ############################################################################
#         # TODO: Initialize weights and biases for the three-layer convolutional    #
#         # network. Weights should be initialized from a Gaussian with standard     #
#         # deviation equal to weight_scale; biases should be initialized to zero.   #
#         # All weights and biases should be stored in the dictionary self.params.   #
#         # Store weights and biases for the convolutional layer using the keys      #
#         # 'theta1' and 'theta1_0'; use keys 'theta2' and 'theta2_0' for the        #
#         # weights and biases of the hidden affine layer, and keys 'theta3' and     #
#         # 'theta3_0' for the weights and biases of the output affine layer.        #
#         ############################################################################
#         # about 12 lines of code
#         self.params['theta1'] = np.random.normal(scale=weight_scale,
#                                                  size=(num_filters, input_dim[0], filter_size, filter_size))
#         self.params['theta2'] = np.random.normal(scale=weight_scale,
#                                                  size=(num_filters * input_dim[1] / 2 * input_dim[2] / 2, hidden_dim))
#         self.params['theta3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
#
#         self.params['theta1_0'] = np.zeros(num_filters)
#         self.params['theta2_0'] = np.zeros(hidden_dim)
#         self.params['theta3_0'] = np.zeros(num_classes)
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         for k, v in self.params.iteritems():
#             self.params[k] = v.astype(dtype)
#
#     def loss(self, X, y=None):
#         """
#         Evaluate loss and gradient for the three-layer convolutional network.
#
#         Input / output: Same API as TwoLayerNet in fc_net.py.
#         """
#
#         theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
#         theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
#         theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
#
#         # pass conv_param to the forward pass for the convolutional layer
#         filter_size = theta1.shape[2]
#         conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
#
#         # pass pool_param to the forward pass for the max-pooling layer
#         pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
#         scores = None
#         ############################################################################
#         # TODO: Implement the forward pass for the three-layer convolutional net,  #
#         # computing the class scores for X and storing them in the scores          #
#         # variable.                                                                #
#         ############################################################################
#         # about 3 lines of code (use the helper functions in layer_utils.py)
#         out_1, cache_1 = conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)
#         out_2, cache_2 = affine_relu_forward(out_1, theta2, theta2_0)
#         scores, cache_3 = affine_forward(out_2, theta3, theta3_0)
#
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         if y is None:
#             return scores
#
#         loss, grads = 0, {}
#         ############################################################################
#         # TODO: Implement the backward pass for the three-layer convolutional net, #
#         # storing the loss and gradients in the loss and grads variables. Compute  #
#         # data loss using softmax, and make sure that grads[k] holds the gradients #
#         # for self.params[k]. Don't forget to add L2 regularization!               #
#         ############################################################################
#         # about 12 lines of code
#         loss, dx = softmax_loss(scores, y)
#         loss += 0.5 * (self.reg * np.sum(theta1 ** 2) + self.reg * np.sum(theta2 ** 2) + self.reg * np.sum(theta3 ** 2))
#
#         dx_3, grads['theta3'], grads['theta3_0'] = affine_backward(dx, cache_3)
#         dx_2, grads['theta2'], grads['theta2_0'] = affine_relu_backward(dx_3, cache_2)
#         dx_1, grads['theta1'], grads['theta1_0'] = conv_relu_pool_backward(dx_2, cache_1)
#
#         grads['theta1'] += self.reg * self.params['theta1']
#         grads['theta2'] += self.reg * self.params['theta2']
#         grads['theta3'] += self.reg * self.params['theta3']
#
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         return loss, grads

# class ConvNet(object):
#   def __init__(self, input_dim=(3, 32, 32), num_filters1=32, num_filters2 = 64, filter_size=7,
#                hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
#     '''
#     [conv-relu-pool]-[conv-relu]-   affine-softmax
#     theta1           theta2         theta3
#     num_filters1     num_filters2
#
#     '''
#     self.params = {}
#     self.reg = reg
#     self.dtype = dtype
#
#     C, H, W = input_dim
#     self.params['theta1'] = np.random.normal(0, weight_scale, (num_filters1, C, filter_size, filter_size))
#     self.params['theta1_0'] = np.zeros(num_filters1)
#     self.params['theta2'] = np.random.normal(0,weight_scale,(num_filters2,num_filters1,filter_size,filter_size))
#     self.params['theta2_0'] = np.zeros(num_filters2)
#     self.params['theta3'] = np.random.normal(0, weight_scale, (num_filters2 * H * W / 4, hidden_dim))
#     self.params['theta3_0'] = np.zeros(hidden_dim)
#     self.params['theta4'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
#     self.params['theta4_0'] = np.zeros(num_classes)
#
#     for k, v in self.params.iteritems():
#       self.params[k] = v.astype(dtype)
#
#   def loss(self, X, y=None):
#     """
#     Evaluate loss and gradient for the three-layer convolutional network.
#
#     Input / output: Same API as TwoLayerNet in fc_net.py.
#     """
#     theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
#     theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
#     theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
#     theta4, theta4_0 = self.params['theta4'], self.params['theta4_0']
#
#     # pass conv_param to the forward pass for the convolutional layer
#     filter_size = theta1.shape[2]
#     conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
#
#
#     # pass pool_param to the forward pass for the max-pooling layer
#     pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
#     scores = None
#     ############################################################################
#     # TODO: Implement the forward pass for the three-layer convolutional net,  #
#     # computing the class scores for X and storing them in the scores          #
#     # variable.                                                                #
#     ############################################################################
#     # about 3 lines of code (use the helper functions in layer_utils.py)
#     # def conv_relu_pool_forward(x, theta, theta0, conv_param, pool_param):
#     out1, cache1 = conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)
#     out2, cache2 = conv_relu_forward(out1, theta2, theta2_0, conv_param)
#     out3, cache3 = affine_relu_forward(out2, theta3, theta3_0)
#     scores, cache4 = affine_forward(out3, theta4, theta4_0)
#
#     ############################################################################
#     #                             END OF YOUR CODE                             #
#     ############################################################################
#
#     if y is None:
#       return scores
#
#     loss, grads = 0, {}
#     ############################################################################
#     # TODO: Implement the backward pass for the three-layer convolutional net, #
#     # storing the loss and gradients in the loss and grads variables. Compute  #
#     # data loss using softmax, and make sure that grads[k] holds the gradients #
#     # for self.params[k]. Don't forget to add L2 regularization!               #
#     ############################################################################
#     # about 12 lines of code
#     loss, dx = softmax_loss(scores, y)
#     loss += 0.5 * self.reg * (np.sum(theta1 ** 2) + np.sum(theta2 ** 2) + np.sum(theta3 ** 2) + np.sum(theta4**2))
#     dx4, dtheta4, dtheta4_0 = affine_backward(dx, cache4)
#     grads['theta4'] = dtheta4+ self.reg*theta4
#     grads['theta4_0'] = dtheta4_0
#     dx3, dtheta3, dtheta3_0 = affine_relu_backward(dx4, cache3)
#     grads['theta3'] = dtheta3 + self.reg * theta3
#     grads['theta3_0'] = dtheta3_0
#     dx2, dtheta2, dtheta2_0 = conv_relu_backward(dx3, cache2)
#     grads['theta2'] = dtheta2 + self.reg * theta2
#     grads['theta2_0'] = dtheta2_0
#     dx1, dtheta1, dtheta1_0 = conv_relu_pool_backward(dx2, cache1)
#     grads['theta1'] = dtheta1 + self.reg * theta1
#     grads['theta1_0'] = dtheta1_0
#
#     ############################################################################
#     #                             END OF YOUR CODE                             #
#     ############################################################################
#
#     return loss, grads


class ConvNet(object):
  def __init__(self, input_dim=(3, 32, 32), num_filters1=32, num_filters2=64, num_filters3=128, num_filters4=256, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
    '''
    [conv-relu-pool] [conv-relu-pool] [conv-relu-pool] - [conv-relu] - [affine-softmax] [affine-softmax]- [affine]
    theta1           theta2             theta3             theta4             theta5       thetha6         theta7
    num_filters      num_filters
    '''
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
#     [conv-relu-pool-1]
    self.params['theta1'] = np.random.normal(0, weight_scale, (num_filters1, C, filter_size, filter_size))
    self.params['theta1_0'] = np.zeros(num_filters1)
#     [conv-relu-pool-2]
    self.params['theta2'] = np.random.normal(0, weight_scale, (num_filters2, num_filters1, filter_size, filter_size))
    self.params['theta2_0'] = np.zeros(num_filters2)
#     [conv-relu-pool-3]
    self.params['theta3'] = np.random.normal(0, weight_scale, (num_filters3, num_filters2, filter_size, filter_size))
    self.params['theta3_0'] = np.zeros(num_filters3)
#     [conv-relu]
    self.params['theta4'] = np.random.normal(0, weight_scale,(num_filters4, num_filters3, filter_size, filter_size))
    self.params['theta4_0'] = np.zeros(num_filters4)
#     [affine-softmax]
    self.params['theta5'] = np.random.normal(0, weight_scale, (num_filters4 * H * W / 64, hidden_dim))
    self.params['theta5_0'] = np.zeros(hidden_dim)
# #     [affine-softmax]
    self.params['theta6'] = np.random.normal(0, weight_scale, (hidden_dim * H * W / 1024, hidden_dim))
    self.params['theta6_0'] = np.zeros(hidden_dim)
#     [affine]
    self.params['theta7'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['theta7_0'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
    theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
    theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
    theta4, theta4_0 = self.params['theta4'], self.params['theta4_0']
    theta5, theta5_0 = self.params['theta5'], self.params['theta5_0']
    theta6, theta6_0 = self.params['theta6'], self.params['theta6_0']
    theta7, theta7_0 = self.params['theta7'], self.params['theta7_0']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = theta1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)
    # def conv_relu_pool_forward(x, theta, theta0, conv_param, pool_param):
    out1, cache1 = conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)
    out2, cache2 = conv_relu_pool_forward(out1, theta2, theta2_0, conv_param, pool_param)
    out3, cache3 = conv_relu_pool_forward(out2, theta3, theta3_0, conv_param, pool_param)
    out4, cache4 = conv_relu_forward(out3, theta4, theta4_0, conv_param)
    out5, cache5 = affine_relu_forward(out4, theta5, theta5_0)
    out6, cache6 = affine_relu_forward(out5, theta6, theta6_0)
    scores, cache7 = affine_forward(out5, theta7, theta7_0)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code
    loss, dx = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(theta1 ** 2) + np.sum(theta2 ** 2) + np.sum(theta3 ** 2) + np.sum(theta4**2) + np.sum(theta5**2) + np.sum(theta6**2) + np.sum(theta7**2))
    dx7, dtheta7, dtheta7_0 = affine_backward(dx, cache7)
    grads['theta7'] = dtheta7 + self.reg*theta7
    grads['theta7_0'] = dtheta7_0
    dx6, dtheta6, dtheta6_0 = affine_relu_backward(dx7, cache6)
    grads['theta6'] = dtheta6 + self.reg * theta6
    grads['theta6_0'] = dtheta6_0
    dx5, dtheta5, dtheta5_0 = affine_relu_backward(dx7, cache5)
    grads['theta5'] = dtheta5 + self.reg * theta5
    grads['theta5_0'] = dtheta5_0
    dx4, dtheta4, dtheta4_0 = conv_relu_backward(dx5, cache4)
    grads['theta4'] = dtheta4 + self.reg * theta4
    grads['theta4_0'] = dtheta4_0
    dx3, dtheta3, dtheta3_0 = conv_relu_pool_backward(dx4, cache3)
    grads['theta3'] = dtheta3 + self.reg * theta3
    grads['theta3_0'] = dtheta3_0
    dx2, dtheta2, dtheta2_0 = conv_relu_pool_backward(dx3, cache2)
    grads['theta2'] = dtheta2 + self.reg * theta2
    grads['theta2_0'] = dtheta2_0
    dx1, dtheta1, dtheta1_0 = conv_relu_pool_backward(dx2, cache1)
    grads['theta1'] = dtheta1 + self.reg * theta1
    grads['theta1_0'] = dtheta1_0

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads







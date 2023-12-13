pass
from cs682.layers import *
from cs682.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b) # fc_cache is same inputted (x, w, b)
                  # fc is "fully connected" layer want to pass it onto the next layer 
    out, relu_cache = relu_forward(a) # ReLU only takes output of the last forward pass
    # and outputs out and relu_cache is 'a' (what was fed in from the output of first layer)
    cache = (fc_cache, relu_cache)  # fc_cache is just original (x, w, b) and relu_cache
    # is just the output of affine_forward pass
    return out, cache # out is output of forward and ReLU. cache is original (x, w, b)
    # and the output of affine_forward pass


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# just throw everything into this function lol
def affine_norm_relu_droupout_forward(X, W, b, gamma, beta, bn_param, norm, use_dropout, dropout_param):
    # X, W, b, gamma, beta, bn_param, norm, dropout, do_param
    """
    Forward pass for the affine-normalization-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache, dropout_cache = 0, 0, 0, 0

    # affine forward pass
    scores, fc_cache = affine_forward(X, W, b)

    # normalization forward pass
    if norm == "batchnorm":
      scores, bn_cache = batchnorm_forward(scores, gamma, beta, bn_param)
    elif norm == "layernorm":
      scores, bn_cache = layernorm_forward(scores, gamma, beta, bn_param)

    # relu forward pass
    scores, relu_cache = relu_forward(scores)

    # dropout
    if use_dropout:
      scores, dropout_cache = dropout_forward(scores, dropout_param)


    # print("poopy forward")

    return scores, (fc_cache, bn_cache, relu_cache, dropout_cache)


def affine_norm_relu_droupout_backward(X, cache, norm, use_dropout):
    """
    Backward pass for the affine-normalization-relu-dropout convenience layer
    """
    dgamma, dbeta = 0, 0
    fc_cache, bn_cache, relu_cache, dropout_cache = cache


    # dropout
    if use_dropout:
      X = dropout_backward(X, dropout_cache)

    # relu backward pass
    dx = relu_backward(X, relu_cache) # or just cache ??

    # normalization backward pass
    if norm == "batchnorm":
      dx, dgamma, dbeta = batchnorm_backward_alt(dx, bn_cache) # or just cache ??
    elif norm == "layernorm":
      dx, dgamma, dbeta = layernorm_backward(dx, bn_cache)

    # affine backward pass
    dx, dw, db = affine_backward(dx, fc_cache)

    # print("poopy backward")

    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

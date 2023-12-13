from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    # should be able to reshape w one line. X = X.reshape(N, d_1 * d_2 ... d_k)
    # you want to get a row of data basically
    # if you have cube of data that's 3 X 4 X 2 then to turn it into a vector of data
    # you'd mult it all together to get it in vector form. Same deal here. Mult 
    # all the d_1...d_k dimensions together to get a vector of data and then
    # assign each vector of data to the N rows of data/N images

    N = x.shape[0] # first dimension of the input data

    # mult all other dimensions together
    # *input_shape feeds all the input shapes in as 1 argument
    # np.prod returns product of an array over a given access
    D = np.prod(x.shape[1:]) # product of d_1...d_k dimensions
    # print(D)
    x_2D = x.reshape(N, D) # shape is (N, D)

    # x_2D = x.reshape(N, -1)  # just need to keep x as N rows

    out = x_2D.dot(w) + b # (N, D) * (D, M) = (N, M) + (M, ) = (N, M)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    # print(x.shape)
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0] # first dimension of the input data
    D = np.prod(x.shape[1:]) # product of d_1...d_k dimensions
    x_2D = x.reshape(N, D)  # (N, D)
    # print(x_2D.shape)

    # now find derivatives of one layer and multiply previous gradient by it

    # what if I just follow the what the dimensions of the outputs are supposed
    # to be ?

    # dx is the backprop of dout -> x which means need to multiply the W 
    # somewhere in there
    # dout (N, M) * w.T (M, D) = (N, D)
    dx = dout.dot(w.T)  # (N, D)

    # dw (D, M)
    # mult times inputs to get weights
    dw = dout.T.dot(x_2D)  # (M, N) * (N, D) = (M, D)
    dw = dw.T # (D, M)

    # db
    # print(dout.shape) # shape is (5, )
    db = np.sum(dout, axis = 0) # error throw is basically that dimension needs
    # to be (5, ) NOT (6, ) which means I'm summing up dout.
    # Yeah def summing up dout that's the same logic as the last hw

    # dx (N, d1, ..., d_k)
    # reshape dx after calc db otherwise it causes dimenstionality issues
    # dx = dx.reshape(N, x_2D.reshape(x.shape))
    # print(dx.shape)
    # print(x.shape)
    dx = dx.reshape(x.shape)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


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
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    
    # derivative of ReLU x = 0 when x < 0
    # thus whenever x < 0, assign dout = 0
    dx = dout.copy()
    dx[x < 0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        # batch mean –> batch variance –> normalize –> scale and shift
        # pg. 3 of Ioffe and Szegedy paper

        # mini-batch mean
        x_mean = np.mean(x, axis = 0)
        
        # mini-batch variance
        x_var = np.var(x, axis = 0)

        # mini-batch normalization
        x_norm = (x - x_mean)/(np.sqrt(x_var + eps))

        # mini-batch scale and shift
        y_scale_shift = gamma * x_norm + beta

        out = y_scale_shift


        # update running mean and running variance with batch mean and var
        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var

        cache = (x, x_mean, x_var, x_norm, gamma, beta, eps, running_mean, running_var)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        # batch mean –> batch variance –> normalize –> scale and shift
        # pg. 3 of Ioffe and Szegedy paper

        # mini-batch mean
        x_mean = bn_param.get('running_mean')
        
        # mini-batch variance
        x_var = bn_param.get('running_var')

        # mini-batch normalization
        x_norm = (x - running_mean)/(np.sqrt(running_var + eps))

        # mini-batch scale and shift
        y_scale_shift = gamma * x_norm + beta

        out = y_scale_shift

        cache = (x, x_mean, x_var, x_norm, gamma, beta, eps, running_mean, running_var)


        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    
    x, x_mean, x_var, x_norm, gamma, beta, eps, running_mean, running_var = cache


    dgamma = np.sum(dout * x_norm, axis = 0)

    dbeta = np.sum(dout, axis = 0)

    # x hat = dout * gamma as notes say
    dx_norm = dout * gamma

    # Derivative of Variance
    dx_var_numerator = -.5 * np.sum(dx_norm * (x- x_mean), axis = 0)
    dx_var_denominator = 1/((x_var + eps) * np.sqrt(x_var + eps))
    dx_var = dx_var_numerator * dx_var_denominator
    # print("dx_var", dx_var)

    # Derivative of Mean
    # dx_meanR = np.sum(dx_norm/np.sqrt(x_var + eps), axis = 0)
    # dx_meanL = dx_var * (2 * np.sum((x - x_mean), axis = 0))/dout.shape[0]
    # dx_mean = (-dx_meanL) + (-dx_meanR)
    dx_meanR = dx_norm/np.sqrt(x_var + eps)
    dx_meanL = dx_var * (2 * (x - x_mean))/dout.shape[0]
    dx_mean = np.sum((-dx_meanL) + (-dx_meanR), axis = 0)
    # print("dx_mean", dx_mean)

    # #derivative of x
    # something's wrong here
    dx = dx_meanR + dx_meanL + dx_mean * 1/dout.shape[0]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, x_mean, x_var, x_norm, gamma, beta, eps, running_mean, running_var = cache

    dx_norm = dout * gamma

    dgamma = np.sum(dout * x_norm, axis = 0)
    dbeta = np.sum(dout, axis = 0)

    dx_numerator = (dout.shape[0] * dx_norm - np.sum(dx_norm, axis = 0) - x_norm * np.sum(dx_norm * x_norm, axis = 0))
    dx = dx_numerator/(dout.shape[0] * np.sqrt(x_var + eps))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
    # print(x.shape)

    # mini-batch mean
    # need to make the dimension (_, 1) for broadcasting when calc x_norm
    x_mean = np.mean(x, axis = 1, keepdims = True)   # axis = 1 ? 
    # print(x_mean.shape)
    
    # mini-batch variance
    # need to make the dimension (_, 1) for broadcasting when calc x_norm
    x_var = np.var(x, axis = 1, keepdims = True)   # axis = 1 ?
    # print(x_var.shape)

    # mini-batch normalization
    x_norm = (x - x_mean)/(np.sqrt(x_var + eps))

    # mini-batch scale and shift
    y_scale_shift = gamma * x_norm + beta

    out = y_scale_shift

    cache = (x, x_mean, x_var, x_norm, gamma, beta, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, x_mean, x_var, x_norm, gamma, beta, eps = cache

    dx_norm = dout * gamma

    dgamma = np.sum(dout * x_norm, axis = 0)
    dbeta = np.sum(dout, axis = 0)

    dx_numerator = (dout.shape[1] * dx_norm - np.sum(dx_norm, axis = 1, keepdims = True) - x_norm * np.sum(dx_norm * x_norm, axis = 1, keepdims = True))
    dx = dx_numerator/(dout.shape[1] * np.sqrt(x_var + eps))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See https://compsci682-fa18.github.io/notes/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

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
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    N, C, H, W = x.shape    # N is # data points. C is # of channels (RGB etc). H and W are input Height and Width
    F, C_filter, HH, WW = w.shape  # HH and WW are filter Height and Width
    
    # Output Dimensions 
    H_prime = int(((H + 2 * pad - HH) / stride) + 1)
    W_prime = int(((W + 2 * pad - WW) / stride) + 1)
    # 3rd dimension is F (# of filters). Final dimension is just H_prime x W_prime x F
    
    # Create a matrix of the N x H_prime x W_prime x F dimensions.
    empty_out = np.zeros((N, F, H_prime, W_prime))  # I will fill in the values of the output matrix

    
    # I'll iterate through N for all the images
    # Then I'll iterate through F bc I want to use all the F filter on all the images.
    # Then I'll iterate by H and by W and calculate the dot products of W.t.dot(x) + b to get that
    # Activation Map. These Activation Maps stack up F (# of filter) times. 
    for n in range(N): # images

        # Pad each image sample
        # np.pad(asdf, 1, constant_values = 0)[1:-1]
        
        # this will output C (# channels) padded channels of the image.
        # Usually 3 (RGB) images that are all padded
        
        x_pad_n = np.pad(x[n], pad_width = pad, constant_values = 0)[1:-1]

        for f in range(F): # filters
            for height_index in range(H_prime):
                for width_index in range(W_prime):    # filters going convolving across the image
                    
                    # current location of height_index
                    h_current = height_index * stride  # multiply by stride to get current index of height_index
                    h_current_edge = h_current + HH
                    # print("h_current", h_current)
                    # print("h_current_edge", h_current_edge)
                    
                    # current location of weight_index
                    w_current = width_index * stride  # multiply by stride to get current index of width_index
                    w_current_edge = w_current + WW
                    # print("w_current", w_current)
                    # print("w_current_edge", w_current_edge)                    

                    # recreate Filter Window 
                    filter_window = x_pad_n[:, h_current:h_current_edge, w_current:w_current_edge]
                    
                    # print("filter_window", filter_window.shape)
                    # print("w[f]", w[f].shape)
                    # print("b[f]", b[f].shape)

                    # Take one of the filters
                    filter_F = w[f, :, :, :] # capturing the fth filter. 
                    # Capture all the channels. Want all the Height and Weight values too bc multiplying
                    # w the entire filter

                    # Matrix Multiplication of Filter Window and Filter for output Activation Map
                    width_out = np.sum(filter_window * filter_F) 

                    # Add bias
                    width_out += b[f]
                    
                    # need to reshape(-1, 1) the filter and multiply it by a reshaped W Weight
                    
                    empty_out[n, f, height_index, width_index] = width_out
                    
                    # empty_out[n, f, h, w] = x_pad_n[______]
      
    out = empty_out
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    

    stride = conv_param['stride']
    pad = conv_param['pad']
    
    N, C, H, W = x.shape    # N is # data points. C is # of channels (RGB etc). H and W are input Height and Width
    F, C_filter, HH, WW = w.shape  # HH and WW are filter Height and Width
    
    # Output Dimensions 
    H_prime = int(((H + 2 * pad - HH) / stride) + 1)
    W_prime = int(((W + 2 * pad - WW) / stride) + 1)

    # Empty Matrices to fill
    dx = np.zeros((N, C, H, W))
    dw = np.zeros((F, C_filter, HH, WW))
    db = np.zeros_like(b)
    dx_interm = np.zeros((N, C, H, W))

    # print("dx", dx.shape) # (4, 3, 5, 5)


    # gotta multiply dout into someting to get dx

    for n in range(N):
      x_pad_n = np.pad(x[n], pad_width = pad, constant_values = 0)[1:-1]
      dx_pad_n = np.pad(dx[n], pad_width = pad, constant_values = 0)[1:-1]
      for f in range(F):
        for height_index in range(H_prime):
          for width_index in range(W_prime):

            h_current = height_index * stride  
            h_current_edge = h_current + HH

            w_current = width_index * stride  
            w_current_edge = w_current + WW
          
            db[f] += dout[n, f, height_index, width_index]

            # recreate Filter Window 
            filter_window = x_pad_n[:, h_current:h_current_edge, w_current:w_current_edge]
            

            # Take one of the filters
            filter_F = w[f, :, :, :] 

            # Matrix Multiplication of Filter Window and Filter for output Activation Map
            width_out = np.sum(filter_window * filter_F)
            
            width_out += b[f]
            
            # db[f] +=  
            
            # print("filter_window", filter_window.shape)
            
            dw[f] += filter_window * dout[n, f, height_index, width_index]
            
            
            # have to multiply current filter_window by the current dout (derivative out) value to get the
            # dw (derivative Weight) value of the current f Filter
            
            # need to multiply current Weight by current dout (derivative out) value to get the
            # dx (derivative x) value of the current f Filter
            dx_pad_n[:, h_current:h_current_edge, w_current:w_current_edge] += w[f] * dout[n, f, height_index, width_index]

            # I'll have to remove the padding before I return dx so it can be accurately backpropagated.
            # Need to remove the borders of padding from each channel. I can start at the pad on the left side
            # when dealing with width.
            
            padded_input_height = pad + H + pad
            padded_input_width = pad + H + pad
            # pad + H + pad - pad = right-hand bound of input
            
            dx_interm[n] = dx_pad_n[:, pad:(padded_input_height - pad), pad:(padded_input_width - pad)]

    dx = dx_interm
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
        
    stride = pool_param['stride']
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    
    N, C, H, W = x.shape    # N is # data points. C is # of channels (RGB etc). H and W are input Height and Width
    
    # Output Dimensions 
    H_prime = int(((H - pool_height) / stride) + 1)
    W_prime = int(((W - pool_height) / stride) + 1)

    # Create a matrix of the N x H_prime x W_prime x C dimensions.
    empty_out = np.zeros((N, C, H_prime, W_prime))  # I will fill in the values of the output matrix

    for n in range(N): # images
      for c in range(C): # Channels
        for height_index in range(H_prime):
          for width_index in range(W_prime):    # filters going convolving across the image
            
            # current location of height_index
            h_current = height_index * stride  # multiply by stride to get current index of height_index
            h_current_edge = h_current + pool_height
            
            # current location of weight_index
            w_current = width_index * stride  # multiply by stride to get current index of width_index
            w_current_edge = w_current + pool_width
          

            # recreate Filter Window 
            filter_window = x[n, c, h_current:h_current_edge, w_current:w_current_edge]

            # Maxiumum of window
            window_max = np.max(filter_window)

            empty_out[n, c, height_index, width_index] = window_max
                    

    out = empty_out
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    x, pool_param = cache

    stride = pool_param['stride']
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    
    N, C, H, W = x.shape    # N is # data points. C is # of channels (RGB etc). H and W are input Height and Width
    
    # Output Dimensions 
    H_prime = int(((H - pool_height) / stride) + 1)
    W_prime = int(((W - pool_height) / stride) + 1)

    # Create a matrix of the N x H x W x C dimensions.
    empty_out = np.zeros((N, C, H, W))  # I will fill in the values of the output matrix

    for n in range(N): # images
      for c in range(C): # Channels
        for height_index in range(H_prime):
          for width_index in range(W_prime):    # filters going convolving across the image
            
            # current location of height_index
            h_current = height_index * stride  # multiply by stride to get current index of height_index
            h_current_edge = h_current + pool_height
            
            # current location of weight_index
            w_current = width_index * stride  # multiply by stride to get current index of width_index
            w_current_edge = w_current + pool_width
          

            # recreate Filter Window 
            filter_window = x[n, c, h_current:h_current_edge, w_current:w_current_edge]

            # Maxiumum of window
            window_max = np.max(filter_window)


            # replacing the window_max with dout[n, c, height_index, width_index]
            filter_window_empty = np.zeros_like(filter_window).flatten()  # zero matrix like filter_window flattened
            window_flatten = filter_window.flatten()  # flatten filter_window so we can get the right index
            filter_window_idx = np.argmax(filter_window)  # index of max value in the filter window
            # window1 = window_flatten[filter_window_idx] # accurately returns max value in filter_window

            filter_window_empty[filter_window_idx] = dout[n, c, height_index, width_index]
            filter_window_empty = filter_window_empty.reshape(pool_height, pool_width)


            empty_out[n, c, h_current:h_current_edge, w_current:w_current_edge] += filter_window_empty

    # my output needs to be same dimensions as X. I need to multiply the derivative of each value in the 
    # pooled output with 
    dx = empty_out
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    
    # calc the metrics w respect to the C Channels
    # so I'm feeding N H W,C = N,D dimensions into batchnorm_forward
    x_spatialBN = np.transpose(x, (0, 2, 3, 1)).reshape(-1, C)
    
    out, cache = batchnorm_forward(x_spatialBN, gamma, beta, bn_param)
    
    out = out.reshape(N, H, W, C)
    out = np.transpose(out, (0, 3, 1, 2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    
    # calc the metrics w respect to the C Channels
    # so I'm feeding N H W,C = N,D dimensions into batchnorm_forward
    dout_spatialBN = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C)
    
    dx1, dgamma, dbeta = batchnorm_backward(dout_spatialBN, cache)
    
    dx2 = dx1.reshape(N, H, W, C)
    dx = np.transpose(dx2, (0, 3, 1, 2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1,C,1,1)
    - beta: Shift parameter, of shape (1,C,1,1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    
    N, C, H, W = x.shape

    x_spatialGN = x.reshape(N, G, int(C/G), H, W)
    
    # I'll be doing computation on C/G, H, and W axes bc we're normalizing
    # per-datapoint per-group

    # mini-batch mean
    x_mean = np.mean(x_spatialGN, axis = (2, 3, 4),keepdims = True)

    # mini-batch variance
    x_var = np.var(x_spatialGN, axis = (2, 3, 4),keepdims = True)

    # mini-batch normalization
    x_norm = (x_spatialGN - x_mean)/(np.sqrt(x_var + eps))

    # reshape so can be broadcasted
    x_norm_reshaped = x_norm.reshape(N, C, H, W)

    # mini-batch scale and shift
    y_scale_shift = gamma * x_norm_reshaped + beta

    out = y_scale_shift

    cache = (x, x_spatialGN, x_mean, x_var, x_norm_reshaped, G, gamma, beta, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1,C,1,1)
    - dbeta: Gradient with respect to shift parameter, of shape (1,C,1,1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    N, C, H, W = dout.shape

    x, x_spatialGN, x_mean, x_var, x_norm_reshaped, G, gamma, beta, eps = cache

    # reshape x to Group Normalization dimensions
    x_GN = x.reshape((N,G,int(C/G),H,W))
    # print("x_GN.shape", x_GN.shape)

    # Calculate dx_norm and reshape it to Group Normalization dimensions
    dx_norm = dout * gamma
    dx_GN = dx_norm.reshape((N,G,int(C/G),H,W))
    # print("dx_GN.shape", dx_GN.shape)


    dbeta = np.sum(dout, axis = (0,2,3), keepdims = True)
    dgamma = np.sum(dout * x_norm_reshaped, axis = (0,2,3), keepdims = True)
    

    # Derivative of the variance
    dvar_scaled = (np.sum(-.5 * dx_GN * (x_GN - x_mean)/(x_var + eps)**1.5, axis = (2,3,4), keepdims = True))/(int(C/G) * H * W)
    # print("dvar_scaled.shape", dvar_scaled.shape)
    # print("dvar_scaled", dvar_scaled)

    # Derivative of the mean
    dmean_numerator = np.sum(x_GN - x_mean,axis = (2,3,4), keepdims = True)
    # print("dmean_numerator.shape", dmean_numerator.shape)
    # print("dmean_numerator", dmean_numerator)
    dmean_scaled = (dmean_numerator * (-2 * dvar_scaled/(int(C/G) * H * W)) + np.sum(-dx_GN/np.sqrt(x_var + eps), axis = (2,3,4), keepdims = True))/(int(C/G) * H * W)
    # print("dmean_scaled.shape", dmean_scaled.shape)

    # Derivative of x
    dx = 2 * dvar_scaled * (x_GN - x_mean) + dmean_scaled + dx_GN/np.sqrt(x_var + eps)
    # print("dx.shape", dx.shape)
    # print("dx", dx)

    dx = dx.reshape(N,C,H,W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

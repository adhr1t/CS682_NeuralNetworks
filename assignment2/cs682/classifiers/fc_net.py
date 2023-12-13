from builtins import range
from builtins import object
import numpy as np

from cs682.layers import *
from cs682.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
          
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg        

        # reshape x's dimensions
        # N = X.shape[0] # first dimension of the input data

        # # mult all other dimensions together
        # # *input_shape feeds all the input shapes in as 1 argument
        # # np.prod returns product of an array over a given access
        # D = np.prod(X.shape[1:]) # product of d_1...d_k dimensions
        # # print(D)
        # x_2D = X.reshape(N, D) # shape is (N, D)


        # affine - relu - affine - softmax

        # do affine and relu forward passes via affine_relu_forward
        out_fc1, cache_fc1 = affine_relu_forward(X, W1, b1) # fc_1 is "fully connected 1"
        # out_fc1 is output of forward and ReLU. cache_fc1 is original (x, w, b)
        # and the output of affine_forward pass
        # print('hello')
        # another affine forward pass
        scores, cache_fc2 = affine_forward(out_fc1, W2, b2)
        # print('hello1')
        # I can call softmax_loss in gradient part bc it calculates the forward loss
        # AND it calculates the backward pass. Just need to work backwards from there
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # softmax-affine
        loss, dscores = softmax_loss(scores, y) # loss is scalar of the loss
        # dscores is gradient of the loss with respect to scores
        # print('hello2', loss)
        # add regularization to loss as you would in forward pass
        L2_reg = .5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
        loss += L2_reg
        # print('hello3')

        # affine backward for second fully-connected layer
        dx2, dW2, db2 = affine_backward(dscores, cache_fc2)
        # print('hello4')
        # store gradients in grads
        grads['W2'] = dW2
        grads['b2'] = db2

        # relu-affine
        dx1, dW1, db1 = affine_relu_backward(dx2, cache_fc1)
        grads['W1'] = dW1
        grads['b1'] = db1

        # regularization to gradients
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims)
        # self.params['b1'] = np.zeros(hidden_dims)
        # self.params['W2'] = weight_scale * np.random.randn(hidden_dims, num_classes)
        # self.params['b2'] = np.zeros(num_classes)

        # needa create some sort of loop that initializes W Weights and b for each
        # of the self.num_layers (the number of hidden layers)

        # for i in range(self.num_layers): # need to go self.num_layers - 1 (bc output
        # # layer doesn't have W Weights), but range() already does - 1
        #   # first layer is just W1. Will feed input_dim and hidden_dims dimensions in
        #   if i == 0:
        #     self.params['W' + str(i+1)] = weight_scale * np.random.randn(input_dim, hidden_dims)
        #   # last layer is W(self.num_layers - 1). Will feed hidden_dims and num_classes dimensions in
        #   elif i == self.num_layers - 1:
        #     self.params['W' + str(i)] = weight_scale * np.random.randn(hidden_dims, num_classes)
        #   # all other layers. Will feed hidden_dims and hidden_dims dimensions in
        #   else:
        #     # this isn't going to work bc when it reaches the last hidden_dims[i], there won't
        #     # be any hidden_dims[i+1] and it will throw an error
        #     self.params['W' + str(i+1)] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i+1])

        # if I can just iterate directly through the diff layer dimensions then I can just
        # iterate from input_dim -> hidden_dims -> num_classes and feed that in directly
        # to the weights instantiation

        # concatenate input_dim, hidden_dims, num_classes horizontally
        # hidden_dims is a frickin list ??????
        # layers_1 = np.array([input_dim, hidden_dims])
        # layers_1 = np.concatenate((input_dim, hidden_dims), axis = 0)
        # layers_all = np.concatenate(layers_1, num_classes)
        layers_all = [input_dim] + hidden_dims + [num_classes]
        self.layers_all = layers_all

        ########### [D] + [H1,H2] + [C] will just add the lists up []

        # layers_all = np.array([input_dim, hidden_dims, num_classes]).ravel()
        # print(layers_all)
        # print(layers_1)

        # i = 0 is input layer which 
        for i in range(len(layers_all)-1):
          self.params['W' + str(i+1)] = weight_scale * np.random.randn(layers_all[i], layers_all[i+1])
          self.params['b' + str(i+1)] = np.zeros(layers_all[i+1])

        # print(self.params.keys()) # has all the right values

        # list(zip(dims[:-1], dims[1:]))
        # For i, (d_in, d_out) in enumerate(list(zip(dims[:-1], dims[1:]))):

        if self.normalization != None:
          for j in range(len(layers_all)-2):
            self.params['gamma' + str(j+1)] = np.ones(layers_all[j+1])
            self.params['beta' + str(j+1)] = np.zeros(layers_all[j+1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # create cache dict to hold all the cache values
        cache_dict = {}
        gamma, beta, bn_param = 0, 0, 0

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # W1, b1 = self.params['W1'], self.params['b1']
        # W2, b2 = self.params['W2'], self.params['b2']
        # reg = self.reg        

        # scores, cache_fc = affine_forward(X, W1, b1) # fc is "fully connected"
        # out_fc1 is output of forward and ReLU. cache_fc1 is original (x, w, b)
        # and the output of affine_forward pass

        # For a network with L layers, the architecture will be
        # {affine - relu} x (L - 1) - affine - softmax
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

        # for k, v in self.params.items():
        #   print(k, v)
        
        # layers_all = np.hstack((input_dim, hidden_dims, num_classes))
        
        # print('2', self.params.keys()) # has all the right values
        # print('3', self.params['W1']) # has all the right values
        
        # print(self.layers_all)
        # # print(self.layers_all[1]) # returns 20 
        # print(len(self.layers_all))
        
        scores = X
        
        # need to iterate through indexes of the concatenated layers
        for i in range(1, len(self.layers_all)): #start at one to ignore input layer
          
          # print('i', i)
          # print(self.layers_all)  # [3072, 100, 100, 100, 100, 100, 10]

          # assign W Weight and b bias of this layer
          # these are confirmed to work properly
          W = self.params['W' + str(i)]
          b = self.params['b' + str(i)]
          
          # if we're on the Lth layer, then no ReLU or normalization.
          # We're doing affine - softmax (last layer)
          if i == len(self.layers_all)-1:
            # print('i - final layer', i, 'out.shape', out.shape, 'W.shape', W.shape, "b.shape", b.shape)
            scores, cache = affine_forward(scores, W, b)
            # print('i', i, 'out.shape', out.shape, 'W.shape', W.shape, "b.shape", b.shape)
            # print('cache', cache)
            cache_dict['cache' + str(i)] = cache
            break


          # affine layer
          # scores_affine, cache_affine = affine_forward(X, W, b)

          # ReLU
          # scores_relu, cache_relu = relu_forward(scores_affine)

          # affine - ReLU layers
          # currently feeding X into every forward pass, which is incorrect.
          # Only the first forward pass should receive X. Need to store the output

          # print('i', i, 'out.shape', out.shape, 'W.shape', W.shape, "b.shape", b.shape)

          # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

          if self.normalization != None:
              gamma = self.params['gamma' + str(i)] # i + 1 ?
              beta = self.params['beta' + str(i)] # i + 1 ?
              # pass self.bn_params[0] to the forward pass for the first batch normalization layer.
              # My iteration starts at 1, so I need to do 'i - 1'
              bn_param = self.bn_params[i - 1] # i ?

          # gamma, beta = None, None
          scores, cache = affine_norm_relu_droupout_forward(scores, W, b, gamma, beta, bn_param, self.normalization, self.use_dropout, self.dropout_param)
          # print('cache', cache)
          # print('cache.shape', cache.shape)
          cache_dict['cache' + str(i)] = cache

          # this reassigns 'out' to the new output POST affine_relu. This updated
          # 'out' will be the right dimension to be fed into future layers

        # print('i', i, 'out.shape', out.shape, 'W.shape', W.shape, "b.shape", b.shape)
        # # print(cache_dict.keys())
        # print("finished forward passes")

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
          # print(scores)
          return scores

        loss, grads = 0.0, {}
        ##############
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # calc loss and derivative using softmax_loss
        # loss, dscores = softmax_loss(out, y) # loss is scalar of the loss
        # # dscores is gradient of the loss with respect to scores

        # # print('hello2', loss)
        # # add regularization to loss as you would in forward pass
        # L2_reg = .5 * self.reg * (np.sum(W1*W1))
        # loss += L2_reg
        # # print('hello3')

        # # affine backward for second fully-connected layer
        # dx, dW, db = affine_backward(dscores, cache)
        # # print('hello4')
        # # store gradients in grads
        # grads['W2'] = dW
        # grads['b2'] = db

        # same approach but reversing through the layer indexes

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

        ### softmax - affine - relu - affine
        ### softmax - affine - (L - 1) X {[dropout] - relu - [batch/layer norm] - affine}

        # calc loss and derivative using softmax_loss
        loss, dscores = softmax_loss(scores, y) # loss is scalar of the loss

                                    # 
        for j in reversed(range(1, len(self.layers_all))): #start at one to ignore input layer

          W = self.params['W' + str(j)] # affine_backward fixes dimensions on its own
          b = self.params['b' + str(j)]
          cache = cache_dict['cache' + str(j)]

          # print('j', j, 'dscores.shape', dscores.shape, 'W.shape', W.shape, "b.shape", b.shape)
          # print('j', j)

          # if we're on the first backward pass after softmax
          # just run the affine_backward pass
          if j == len(self.layers_all) - 1:
            # print('j - first backprop layer', j, 'dscores.shape', dscores.shape, 'W.shape', W.shape, "b.shape", b.shape)
            dscores, dW, db  = affine_backward(dscores, cache)
            # print("in first affine after softmax", j)

          else:
            ##### CREATE A HELPER FUNCTION IN LAYER_UTILS.PY THAT DOES AFFINE_NORM_RELU_BACKWARD
            # running affine_relu for all other layers
            dscores, dW, db, dgamma, dbeta  = affine_norm_relu_droupout_backward(dscores, cache, self.normalization, self.use_dropout)
            # print("poopy in fc_net")
            # print("in affine_norm_relu_droupout_backward", j)


            # this is messing with my output
            # I was saving it to self.params not grads ! Solved one issue 
            if self.normalization != None: #  and j < len(self.layers_all)
              grads['gamma' + str(j)] = dgamma
              # print("in self.normalization != None", j)
              # if j == 1: print("##################", dgamma)
              grads['beta' + str(j)] = dbeta



          grads['W' + str(j)] = dW
          grads['b' + str(j)] = db  
            
          grads['W' + str(j)] += self.reg * W

          L2_reg = .5 * self.reg * (np.sum(W*W))
          loss += L2_reg


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # print("poopy at end of fc_net loss grads['gamma6']", grads['gamma5'])
        # print("grads at end of loss function", grads)

        return loss, grads

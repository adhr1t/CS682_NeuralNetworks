from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    # print("X.shape", X.shape)
    # print("D", D)

    h1 = X.dot(W1) + b1  # first hidden layer activation (1)
    
    # ReLU activation layer
    # need to do np.maximum bc otherwise this is just another linear predictor, not NN
    # activation function here is ReLU bc I'm doing np.maximum(0, _) with the h1 hidden layer
    # and max(0, _) is the function for ReLU
    a_ReLU1 = np.maximum(0, h1) # (2)

    # output layer/scores
    scores = a_ReLU1.dot(W2) + b2  # (3)
    # print("scores.shape", scores.shape)
    # print("scores", scores)
    # (N, C) (5, 3)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    f = scores
    f -= np.max(f, axis = 1).reshape(-1, 1)   # (N, C) (5, 3)
    

    ######## Steps for Loss Calculation
    exp_f = np.exp(f)  # (4)
    # print("exp_f.shape", exp_f.shape)
    # print("exp_f", exp_f)

    den = np.sum(exp_f, axis = 1, keepdims = True) # (5)
    # print("den.shape", den.shape)
    # print("den", den)

    invden = 1/den  # (6)
    # print("invden.shape", invden.shape)
    # print("invden", invden)

    ## THIS IS JUST THE RAW OUTPUT
    softmax_func = exp_f * invden

    # calc correct_class_scores matrix. This gets 0 – 499 indexes for the 500
    # rows of the 500 images. y is the correct classes. This stores all of 
    # the correct_class_scores in a matrix by essentially matching each image score
    # with the correct classification 0 – 9 and extracting that score
    correct_class_scoreS = softmax_func[np.arange(N), y]
    # print("correct_class_scoreS.shape", correct_class_scoreS.shape) # (5, ) this makes sense bc it's just the 
    #                                                                 # correct class scores of the 5 classes
    # print("correct_class_scoreS", correct_class_scoreS)

    # print("softmax_func.shape[0]", softmax_func.shape[0], "N", N)


    logAll = np.log(correct_class_scoreS) # (7)
    # print("logAll.shape", logAll.shape)
    # print("logAll", logAll)

    sumNeg = -np.sum(logAll)  # (8)

    sumNeg /= N

    loss = sumNeg
      

    # loss = -np.sum(np.log(np.exp(correct_class_scoreS)/ np.sum(np.exp(f), axis = 1))) 
    # loss /= N

    L2_reg = reg * (np.sum(W1*W1) + np.sum(W2*W2))  # (np.sum(W1*W1) + np.sum(W2*W2)) ?

    loss += L2_reg
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    # Backpropagation aka derivative calculation


    # SOFTMAX_FUNC IS THE RAW OUTPUTS OF FORWARD PASS. WE ONLY WANT OUTPUTS
    # FOR DERIVATIVE CALCULATION
    
    # First reverse step is last step of vectorized softmax. We want to subtract 1 whenever
    # the correct class is hit. Indicator function of correct class
    dsumNeg = softmax_func # shape is (N, C)
    # print("dsumNeg.shape1", dsumNeg.shape)

    # matrix of shape f. We want to subtract 1 from dsumNeg whenever
    # the correct class is hit. Indicator function of correct class    
    dsumNeg_correct_classes_ones_matrix = np.zeros_like(dsumNeg)
    dsumNeg_correct_classes_ones_matrix[np.arange(dsumNeg_correct_classes_ones_matrix.shape[0]), y] = 1

    dsumNeg = dsumNeg - dsumNeg_correct_classes_ones_matrix

    # print("dsumNeg.shape2", dsumNeg.shape)
    dsumNeg /= N  # shapes are (N, C) (5, 3)
    # print("dsumNeg.shape3", dsumNeg.shape)
    # print("dsumNeg", dsumNeg)


    # print("W2.shape", W2.shape) # shape is (10, 3)  (W, C)


    # backprop on f = scores = a_ReLU1.dot(W2) + b2
    # except f = sumNeg now after Numerical Stabilization and softmax loss application
    da_ReLU1 = dsumNeg.dot(W2.T) # (3)  # shapes are (N, W) (5, 10)
    # print("da_ReLU1.shape", da_ReLU1.shape)

    # backprop on a_ReLU1
    # a_ReLU1 = np.maximum(0, h1) where h1 = X.dot(W1) + b1
    # thus a_ReLU1 = np.maximum(0, X.dot(W1) + b1)
    dh1 = 1 * da_ReLU1 # (2)
    # print("dh1.shape", dh1.shape)

    # backprop on dh1
    # h1 = X.dot(W1) + b1
    dx = dsumNeg.dot(W2.T)  # (1) # shapes are (N, H) (5, 4)
    dx[h1 <= 0] = 0
    # print("dx.shape", dx.shape)


    # Gradient calculation
    # calc gradients backwards to forwards bc backpop so start with b2 –> W2 –> b1 –> W1

    # im loosing my minddddddddd asdiuhifblaksdl

    b2_grad = np.sum(dsumNeg, axis = 0) # shapes are (C, )  (3, )
    # print("b2_grad.shape", b2_grad.shape)
    # W2_grad = dsumNeg.dot(W2.T) # shape is (5, 10) (N, W)
    W2_grad = a_ReLU1.T.dot(dsumNeg) # shape is (10, 3) (W, C). Looks alot better bc number 
                                     # of gradients matches with weights and correct 
                                     # number of classes too

    # # print("W2_grad.shape", W2_grad.shape)    
    b1_grad = np.sum(dx, axis = 0)
    W1_grad = X.T.dot(dx)


    # # Gradient storage
    # # updating gradients backwards to forwards so start with b2 –> W2 –> b1 –> W1
    grads['b2'] = b2_grad   # adding this bias to all nodes in the gate as the notes say
    # # print("grads['b2'].shape1", grads['b2'].shape)
    # # grads['b2'] = grads['b2'][:, np.newaxis].T
    # # print("grads['b2'].shape2", grads['b2'].shape)

    grads['W2'] = W2_grad

    grads['W2'] += 2 * reg * W2 # regularization. W2 is shape (10, 3). Need to add
    # # b2_grad_add = 2 * reg * W2 + grads['b2']
    # # grads['b2'] = b2_grad_add

    grads['b1'] = b1_grad  # adding this bias to all nodes in the gate as the notes say
    # # print("grads['b1'].shape1", grads['b1'].shape)

    # # grads['b1'] = grads['b1'][:, np.newaxis].T
    # # print("grads['b1'].shape2", grads['b1'].shape)

    grads['W1'] = W1_grad # dx is the final backprop step before we reach the input gate/nodes

    # # b1_grad_add = 2 * reg * W1.T + grads['b1']
    # # grads['b1'] = b1_grad_add

    grads['W1'] += 2 * reg * W1 # regularization
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indices = np.random.choice(num_train, batch_size)
      X_batch = X[indices]
      y_batch = y[indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] += (-learning_rate) * grads['W1']
      self.params['b1'] += (-learning_rate) * grads['b1']
      self.params['W2'] += (-learning_rate) * grads['W2']
      self.params['b2'] += (-learning_rate) * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    h1 = np.maximum(0, X.dot(self.params['W1']) + self.params['b1']) # ReLU activation
    scores = h1.dot(self.params['W2']) + self.params['b2']

    y_pred = np.argmax(scores, axis = 1) # this thing is definitely working
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred



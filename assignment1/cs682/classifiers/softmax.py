import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1] # = 10 (number of classes)
  num_train = X.shape[0] # 49000 training images

  for i in range(num_train):
    scores = X[i].dot(W)  # iteratively multiply weights with each image values
                          # to get each image's 10 class scores. Both X[i] and W are 1D
                          # shape will be (1, 3073) * (3073, 10) = (1, 10)
    
    # Numeric Stabilization by shifting values in scores/f so the max value is 0
    f = scores
    f -= np.max(f)

    correct_class_score = f[y[i]]  # y[i] is correct class's score

    # calculate the numerically stabilized loss for each ith training example
    # feed f_y_i into first slot bc that's what the equation says. f_y_i = 
    # f[y[i]] = correct_class_score
    margin = -np.log(np.exp(correct_class_score) / np.sum(np.exp(f)))
    # print("margin", margin)
    # print("margin.shape", margin.shape)

    loss += margin
    # print("loss1", loss)
    # print("loss.shape1", loss.shape)

    for j in range(num_classes): # iterate through all 10 classes
    # what if I just added all the loss-contributing classes (X[i]) and then
    # subtracted the correct class (y[i]) column (X[i]) from the entire thing ??
    # That worked ! This route actually feels simpler than the route we're provided
    # in the SVM. Easier for my brain to understand

      # get matrix of stabilized images of jth class
      # We're doing this to essentially revert the Numerically Stabilized
      # scores of jth class to smomething that can be divded and added
      # to the dW gradient matrix
      stabilized_images = np.exp(f[j]) * X[i] # (500, 1) * (1, 3073) = (500, _)
      # divide stabilized images f jth class by all stabilized images as stated by
      # softmax function. 
      dW[:, j] += (stabilized_images) / np.sum(np.exp(f))

      
    # if j == y[i]: # if the correct class's index is the same the current jth class/index,
    
    # when we're on the correct class, subtract X[i] as stated by gradient formula
    dW[:, y[i]] -= X[i]
        # continue    
      


  loss /= num_train
  # print("loss2", loss)
  # print("loss.shape2", loss.shape)

  loss += reg * np.sum(W * W)
  # loss = loss.reshape(-1, 1)
  # print("loss3", loss)
  # print("loss.shape3", loss.shape)

  dW /= num_train

  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1] # = 10 (number of classes)
  num_train = X.shape[0] # 49000 or 500 training images

  # calc scores
  scores = X.dot(W) # size is (49000, 10) or (500, 10)

  # Numerically Stabilize scores
  f = scores
  # f = f - np.max(f, axis = 1)
  # print(f.shape)
  # print(np.max(f, axis = 1).shape)
  # print(np.max(f, axis = 1).reshape(500, 1).shape)
  f -= np.max(f, axis = 1).reshape(-1, 1)  # need to do axis = 1 so we specify that
  # we want the  maximum of each row/image

  # calc correct_class_scores matrix. This gets 0 – 499 indexes for the 500
  # rows of the 500 images. y is the correct classes. This stores all of 
  # the correct_class_scores in a matrix by essentially matching each image score
  # with the correct classification 0 – 9 and extracting that score
  correct_class_scoreS = f[np.arange(f.shape[0]), y]

  # with scores I can calc the loss
  # loss = -(correct_class_scores) + np.log(np.sum(scores))
  loss = -np.sum(np.log(np.exp(correct_class_scoreS)/ np.sum(np.exp(f), axis = 1))) # need to do axis = 1
  # so we specify that we want the sum of each row/image like Softmax equation states.
  # Sum entire thing and make it negative bc it'll be a matrix otherwise


  loss /= f.shape[0]

  loss += reg * np.sum(W * W)


  ## Gradient calc

  gradients_not_subtracted = np.zeros_like(f)
  stabilized_images = np.exp(f) # (500, 1) * (1, 3073) = (500, _)
  # divide stabilized images f by all stabilized images as stated by
  # softmax function. 
  gradients_not_subtracted += (stabilized_images) / (np.sum(np.exp(f), axis = 1)).reshape(-1,1)
  # specify axis = 1 so we're
  # summing by row/image like Softmax formula states

  # matrix of shape f. We want to subtract 1 from gradients_not_subtracted whenever
  # the correct class is hit. Indicator function of correct class
  correct_classes_ones_matrix = np.zeros_like(f)
  correct_classes_ones_matrix[np.arange(correct_classes_ones_matrix.shape[0]), y] = 1

  gradients_subtracted = gradients_not_subtracted - correct_classes_ones_matrix

  dW += X.T.dot(gradients_subtracted)

  # full loss of dataset is total loss over all training examples


  dW /= num_train

  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


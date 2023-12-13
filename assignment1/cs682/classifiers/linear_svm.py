import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights. C = 10
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # = 10 (number of classes)
  num_train = X.shape[0] # 49000 training images
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)  # iteratively multiply weights with each image values
                          # to get each image's 10 class scores. Both X[i] and W are 1D
    correct_class_score = scores[y[i]]  # y[i] is correct class's score
    # for this label
    for j in range(num_classes): # iterate through all 10 classes
      if j == y[i]: # if the current class is the same the jth class,
        continue    # skip the rest of the j loop bc no point in comparing 
                    # correct class w itself

      # calc the margin/single-class-loss of this specific jth class. If it's greater than
      # zero, then we must add that to the total loss. If it's less than zero,
      # we're happy with the prediction and don't add anything to the loss

      # scores[j] is all the ith image's scores of class j 
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # print("margin", margin)
      # print("margin.shape", margin.shape)
      if margin > 0:
        loss += margin
        
        # do gradient calcs here where loss is being added
        # the inputted values will be i and j
        # gradient is literally just derivative of a function that
        # takes a vector as an input

        # how  much need to change the W Weight to make the loss smaller
        
        # summing up X_i in dW
        # X[i] is a column, so want to add X[i] into dW
        # in the column dimension [:, j]
        dW[:, j] += X[i]
        # when it corresponds to the correct class (y[i]), we subtract X[i]
        dW[:, y[i]] -= X[i]

        # need another line that deals w class label of X[i] aka y[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # want gradient to be average over all training examples too
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # print("loss2", loss)
  # print("loss.shape2", loss.shape)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  # Gradient: sum of all data points + regularization term
  # dW indicates derivative so is it the tangent of some curve ?? like in the
  # class notes ??
  # This is supposed to be Analytical Gradient. Which was def in the notes
  # Lecture 4 slide 15
  
  # basically I'm taking the derivative of a function, substituting in the 
  # respective pixel values and assigning the resulting vector to dW

  # W matrix - gradient matrix

  
  # Add regularization to the gradient too
  dW += 2 * reg * W

  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  num_train = X.shape[0] # X is 500 training images. 500 x 3073 (500, 3073)
                         # y is 500 training labels. 500 x _ (500, )

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # broadcast subtract scores of scores - correct class scores of all other classes + 1

  scores = X.dot(W) # X is (500, 3073) * (3073, 1) = (500, _)

  # with the scores matrix, we need to go through and subtract 
  # scores - correct class scores of all other classes + 1. Do this for
  # all 10 "correct classes"

  # calc correct_class_scores matrix. This gets 0 – 499 indexes for the 500
  # rows of the 500 images. y is the correct classes. This stores all of 
  # the correct_class_scores in a matrix by essentially matching each image score
  # with the correct classification 0 – 9 and extracting that score
  correct_class_scores = scores[np.arange(scores.shape[0]), y] # np.arange might work better
  # print("correct_class_scores 2", correct_class_scores.shape)

  # delta is a matrix of same size of scores full of 1s bc they all needed
  # to be added to the scores - correct class scores

  #########
  # delta = np.ones(scores.shape)
  # print("loss margin 2", loss_margin.shape)


  # loss_margin = scores.T - correct_class_scores.T + delta.T
  loss_margin = scores.T - correct_class_scores.T + 1 # don't need a delta matrix. Can just add 1 and
  # it'll add the scalar 1 to everything

  # print("loss margin 1", loss_margin.shape)

  # if any value in matrix loss_m is < 0, replace it with 0
  # loss_m = loss_m < 0
  loss_margin[loss_margin < 0] = 0
  loss_margin[y, np.arange(scores.shape[0])] = 0 # np.arange might work better
  
  loss_margin = loss_margin.T
  # print("loss margin 2", loss_margin.shape)


  loss = np.sum(loss_margin)
  
  loss /= num_train

  loss += reg * np.sum(W * W)

  loss_margin_copy = loss_margin.copy()
  loss_margin_copy[loss_margin_copy > 0] = 1
  # print("loss margin 3", loss_margin.shape)

  # we know if a certain margin adds to the loss (by being > 0) bc it's
  # now labeled w 1. We now sum all those margins up
  sum_loss_margin = np.sum(loss_margin_copy, axis = 1) # not summing on right axis ???
  # print("sum loss margin 1", sum_loss_margin.shape)

  # assigning sum_loss_margin vals to a negative version of itself
  # bc it's in the correct class, as the notes say
  loss_margin_copy[np.arange(scores.shape[0]), y] = -sum_loss_margin 
  # print("loss margin 4", loss_margin.shape)

  # print("X 1", X.shape)

  dW += X.T.dot(loss_margin_copy) # X needs to be (3073, 500) to be multiplied into (500, _) 

  dW /= num_train
  
  dW += 2 * reg * W


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

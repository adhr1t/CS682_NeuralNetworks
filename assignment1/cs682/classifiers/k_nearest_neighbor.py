import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """

    num_test = X.shape[0]
    #print(num_test)
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #print(dists.shape)
    for i in range(num_test):
      #print("i", i, ":", list(X[i]))
      for j in range(num_train):
        #print("j", j, ":", self.X_train[j].shape)
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        # distances = np.sqrt(np.sum(np.square(np.asarray(list(X[i])) - np.asarray(list(self.X_train[j])), axis = 1)))
        
        # here it subtracts ith test example minus jth train example pixels, squares all the
        # differences of pixel values, sums them up, and square roots them for 
        # that specific ith test vs jth train example image in the dist matrix
        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        
        # print('dists[i, j]', dists[i, j], 'j', j, 'num vals', dists[i, j].size)

        
        #print(distances)
        # print("hello potato", dists)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
      #print("two loops", dists[i, :])
    return dists

    # print('hello potato')
    # return 0

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    # print(dists.shape)
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      
      # the calculation is getting amalgamated. The difference between 
      # the ith test example vs each train example is correct. Then
      # they all get squared which is also correct. Then I want to 
      # sum each individual pixel difference, but this equation sums every single
      # thing going on in the 
      
      # print("self.X_train", self.X_train.shape)
      # print("np.square(X[i] - self.X_train[i,:])", (np.square(X[i] - self.X_train[i,:])).shape)
      

      ## this proves that all 3072 pixel differences are being computed
      # diff_squared = np.square(X[i] - self.X_train[i,:])
      # print(diff_squared.shape)
      # print('diff_squared', diff_squared, 'i', i, 'num vals', len(diff_squared))
      
      # dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train[i,:])))  # correct math
      # dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train[i,:]), axis = 1))
      dists[i, :] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train), axis = 1))
      # dists[i, :] = np.sqrt((np.square(X[i] - self.X_train[i,:])))
      # print("one loop",dists[i, :])
      
      # min_index = np.argmin(distances) # get the index with smallest distance
      # dists[i] = self.ytr[min_index] # predict the label of the nearest example
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]

    # print(X.shape)
    # print(self.X_train.shape)

    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    
    # just subtract the test images from all the train images
    # difference = X - self.X_train

    # break down the problem and do it piecewise bc there isn't an iterative 
    # factor to use anymore

    # I can square and sum the training and testing images prior and 
    # find their difference later
    X_train_processed = np.sum(np.square(self.X_train), axis = 1)
    X_test_processed = np.sum(np.square(X), axis = 1)

    # reorient
    # X_train_processed = X_train_processed[: ,np.newaxis]
    # X_test_processed = X_test_processed[np.newaxis, :]

    X_train_processed = X_train_processed[np.newaxis, :]
    X_test_processed = X_test_processed[: ,np.newaxis]

    # matrix multiplation means the dot product 
    # i can do dot product if I transpose one of the image matricies bc 
    # they have the same 3072 dimension
    dot_X_train_test_processed = np.dot(X, self.X_train.T)
    dot_X_train_test_processed = -2 * dot_X_train_test_processed

    # with dimensions oriented the diff componentes 
    # can be combined and square rooted and assigned to dist
    dists = np.sqrt(X_train_processed + X_test_processed + dot_X_train_test_processed)

    # print(dot_X_train_test_processed.shape)   # shape (5000,500)
    # need to get orientation same
    # X_train_processed currently (5000,) and X_test_processed currently (500,)
    # X_test_processed needs have newaxis or be transposed so it's (,500)
    # to match dot_X_train_test_processed





    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test): # this is going down the rows of test images
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort. 
      #             'Returns the indices that would sort an array.'           #
      #########################################################################
      
      # dists[i] is every row of test image vs training image
      distances_ordered = np.argsort(dists[i])

      # capture the indexes of the 'k' smallest distances. 
      # Basically capture the first 'k' values in the sorted array
      # this finds the k nearest neighbors of the ith testing point
      k_closest = distances_ordered[:k] # these are their INDEXES
      # they're the indexes of the training values too. So I should be able
      # to locate the training labels with just their indexes
      
      ## testing
      # import numpy as np
      # distances_ordered = np.argsort([3, 2, 8, 5, 6])
      # k_closest = distances_ordered[:3]

      # print("k_closest", k_closest)

      # print("self.y_train", self.y_train)

      # # this outputs the label number 0 – 10
      # print("self.y_train[k_closest]", self.y_train[k_closest])

      closest_y.append(self.y_train[k_closest])

      # print(closest_y) # only exists in the loop
      # img_label = self.y_train


      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      
      # values is sorted ascending and duplicates removed
      # counts is the corresponding frequencies to each value
      values, counts = np.unique(closest_y, return_counts=True)
      
      # get counts.max() to get maximum frequency. Then iterate through
      # counts and if any value matches counts.max(), I store it in maxes []
      # and store the respective nth value in values in popular_labels []. 
      # This will get me the greatest frequencies and their respective labels.
      # From there I can do y_pred[i] = popular_labels.min()
      
      max_freq = counts.max()
      popular_labels = []

      for n in range(len(counts)):
        # if one of the frequencies equals the maximum frequencies,
        # I know it corresponds to one of the most popular labels
        if counts[n] == max_freq:
          # I store the popular label in popular_labels list
          popular_labels.append(values[n])

      y_pred[i] = np.asarray(popular_labels).min()

      # use np.argsort to organize counts (so max)


      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred


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
    num_train = x.shape[0]

    correct_class_scores = x[range(num_train), y].reshape(num_train, 1)  # (N, 1)
    margins = np.maximum(0, x - correct_class_scores + 1)  # (N, C)
    margins[range(num_train), y] = 0  # margins[yi, yi] = 0
    loss = np.sum(margins) / num_train

    dx = (margins > 0).astype(int)  # (N, C)
    dx[range(num_train), y] = -np.sum(dx, axis=1)
    dx = dx / num_train

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
    num_train = x.shape[0]

    scores = x - np.max(x, axis=1).reshape(-1, 1)

    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
    loss = -np.sum(np.log(softmax[range(num_train), list(y)]))
    loss = loss / num_train

    dx = softmax.copy()
    dx[range(num_train), list(y)] += -1
    dx = dx / num_train

    return loss, dx

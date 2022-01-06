from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.reshape(x,(x.shape[0],-1)).dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

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
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    db = np.sum(dout,axis=0)
    dw = ((np.reshape(x,(x.shape[0],-1))).T).dot(dout)
    dx = dout.dot(w.T)
    dx = np.reshape(dx,x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x,0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout*((x>0)*1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """
    stops overflow but idk how
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    """

    probs = np.exp(x)
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

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
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # All operation is element wise broadcasting
        # (D,)operation(N,D) -> valid -> column wise element operation
        # (D,)operation(D,N) -> Not valid
        # (D,1)operation(D,N) -> valid -> element wise row operation

        sample_mean = np.mean(x, axis=0) # (D,)(1 / N) * np.sum(x, axis=0)

        x_mean = x - sample_mean # (N,D)

        sample_var = np.var(x,axis=0) #(D,)(1 / N) * np.sum(((x - mu) ** 2), axis=0)

        invert_var = 1/np.sqrt(sample_var + eps)# (D,)

        x_norm = x_mean*invert_var # (N,D)

        out = gamma*x_norm + beta # (D,)*(N,D) + (D,)->(N,D) 

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = { 
          "XNorm" : x_norm,
          "Gamma" : gamma,
          "XMean" : x_mean,
          "Inverted_Variance" : invert_var
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html -> referece

    N,D = dout.shape

    # step 9
    dbeta = np.sum(dout,axis=0)

    # step8
    dgamma = np.sum(dout*cache["XNorm"],axis=0)
    dx_norm = dout*cache["Gamma"]

    # step 7
    dinvar = np.sum(dx_norm*cache["XMean"],axis=0)
    dx_mean1 = dx_norm*cache["Inverted_Variance"]

    # step 6
    dsqrtvar = -1*dinvar*(cache["Inverted_Variance"]**2)

    # step 5
    dvar = 0.5*cache["Inverted_Variance"]*dsqrtvar

    # step 4
    dsq = 1. /N * np.ones((N,D)) * dvar

    # step 3
    dx_mean2 = 2*cache["XMean"]*dsq

    # step 2
    du = np.sum(dx_mean1+dx_mean2,axis=0)*-1
    dx1 = dx_mean1 + dx_mean2

    # step 1
    dx2 = (1/N) * np.ones((N,D))*du
    
    # step 0
    dx = dx1 +dx2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://kevinzakka.github.io/2016/09/14/batch_normalization/ -> reference
    
    invert_var,x_norm,gamma = cache["Inverted_Variance"],cache["XNorm"],cache["Gamma"]

    N = x_norm.shape[0]

    dbeta = np.sum(dout,axis=0)

    dgamma = np.sum(dout*x_norm,axis=0)

    dx_norm = dout*gamma
    
    dx = (1/N)*invert_var*(N*dx_norm - np.sum(dx_norm,axis=0) - x_norm*np.sum(x_norm*dx_norm,axis = 0))
    
    # Sanity Check
    # invert_var -> (D,)
    # dx_norm -> (N,D)
    # np.sum(dx_norm,axis=0) -> (D,)
    # x_norm*dx_norm -> (N,D)
    # np.sum(x_norm*dx_norm,axis = 0) -> (N,D)
    # x_norm*np.sum(x_norm*dx_norm,axis = 0) -> (N,D)*(D,) : (N,D)
    # All element wise multiplication

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

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
    eps = ln_param.get("eps", 1e-5)
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sample_mean = np.mean(x, axis=1, keepdims=True) # (D,)(1 / N) * np.sum(x, axis=0)

    x_mean = x - sample_mean # (N,D)

    sample_var = np.var(x,axis=1, keepdims=True) #(D,)(1 / N) * np.sum(((x - mu) ** 2), axis=0)

    invert_var = 1/np.sqrt(sample_var + eps)# (N,1)

    x_norm = x_mean*invert_var # (N,D)

    out = gamma*x_norm + beta # (D,)*(N,D) + (D,)->(N,D) 

    cache = { 
      "XNorm" : x_norm,
      "Gamma" : gamma,
      "XMean" : x_mean,
      "Inverted_Variance" : invert_var
    }

    """
      can use transpose and calculate
    """

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    invert_var,x_norm,gamma = cache["Inverted_Variance"],cache["XNorm"],cache["Gamma"]

    D = x_norm.shape[1]

    dbeta = np.sum(dout,axis=0)

    dgamma = np.sum(dout*x_norm,axis=0)

    dx_norm = dout*gamma
    
    dx = (1/D)*invert_var*(D*dx_norm - np.sum(dx_norm,axis=1,keepdims=True) - x_norm*np.sum(x_norm*dx_norm,axis = 1,keepdims=True))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

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
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        mask = ( np.random.rand(*x.shape) < p ) / p
        out = x * mask
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpacking all neccesaary things 
    N,C,H,W = x.shape
    pad, stride = conv_param['pad'],conv_param['stride']
    F,_,HH,WW = w.shape
    
    assert (H + 2 * pad - HH)%stride == 0
    assert (W + 2 * pad - WW)%stride == 0
    Hout = 1 + (H + 2 * pad - HH) // stride #H'
    Wout = 1 + (W + 2 * pad - WW) // stride #W'

    # create output tensor
    out = np.zeros((N,F,Hout,Wout))

    # padding the input
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')

    """
    # Naive Loops 
    for n in range(N):
      for f in range(F):
        for w1 in range(Wout):
          for h in range(Hout):
            out[n,f,h,w1] = np.sum(  x_pad[ n, :, (h*stride):(h*stride + HH), (w1*stride):(w1*stride + WW)] * w[f,:,:,:]  ) + b[f]  

    """
    # Efficient Matrix solution

    # reshaping w into column vectors of f filters
    w_row = w.reshape((w.shape[0],-1))

    # declaring x_col
    x_col = np.zeros((HH*WW*C , Hout*Wout))

    for n in range(N):
      neuron = 0
      for h in range(Hout):
        for w1 in range(Wout):
          x_col[:,neuron] = x_pad[ n, :, (h*stride):(h*stride + HH), (w1*stride):(w1*stride + WW)].reshape(HH*WW*C)
          neuron +=1
      out[n] = (np.dot(w_row,x_col) + b.reshape((F,1))).reshape((F,Hout,Wout))
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
    # https://www.youtube.com/watch?v=pUCCd2-17vI

    
    x,w,b,conv_param = cache

    N,C,H,W = x.shape
    pad, stride = conv_param['pad'],conv_param['stride']
    F,_,HH,WW = w.shape

    db = np.zeros(F)
    dw = np.zeros(w.shape)
    dx = np.zeros_like(x)

    assert (H + 2 * pad - HH)%stride == 0
    assert (W + 2 * pad - WW)%stride == 0
    Hout = 1 + (H + 2 * pad - HH) // stride #H'
    Wout = 1 + (W + 2 * pad - WW) // stride #W'

    # inverse the matrix
    w_ = np.zeros_like(w)
    for i in range(HH):
        for j in range(WW):
            w_[:,:,i,j] = w[:,:,HH-i-1,WW-j-1]

    # padding the input
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')

    # padding dout
    y_pad = np.pad(dout,((0,0),(0,0),(WW-1,WW-1),(HH-1,HH-1)), 'constant')

    # dx_pad
    dx_pad = np.zeros_like(x_pad)

    for f in range(F):
      db[f] = np.sum(dout[:,f,:,:])
    """ 
      for c in range(C):
        for h in range(HH):
          for w1 in range(WW):
            dw[f,c,h,w1] = np.sum( dout[:, f, :, :]*x_pad[:, c, (h*stride):(h*stride + Hout), (w1*stride):(w1*stride+Wout)])


    """
    for n in range(N):
        for f in range(F):
            for j in range(0, Hout):
                for i in range(0, Wout):
                    dw[f] += x_pad[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW]*dout[n,f,j,i]  
    
    """
    # Version without vectorisation
    for n in range(N):       # for all examples
        for f in range(F):   # for all filters
            for i in range(H+2*pad): # for all indexes in x_pad
                for j in range(W+2*pad):
                    for k in range(HH): # for all indexes of w180 degree rotated
                        for l in range(WW):
                            for c in range(C): # profondeur
                                dx_pad[n,c,i,j] += y_pad[n, f, i+k, j+l] * w_[f, c, k, l]
    
    """
    # Better solution : Not mine need to see what is the logic
    for n in range(N):
        for f in range(F):
            for j in range(0, Hout):
                for i in range(0, Wout):
                    dx_pad[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW] += w[f]*dout[n, f, j, i]
    
    dx = dx_pad[:,:,pad:pad+H,pad:pad+W]   
    """

    # extract params 
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    # pad H and W axes of the input data, 0 is the default constant for np.pad
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # output volume size
    # note that the // division yields an int (while / yields a float)
    Hout = (H + 2 * pad - HH) // stride + 1 
    Wout = (W + 2 * pad - WW) // stride + 1

    # construct output
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # naive Loops
    for n in range(N): # for each neuron
        for f in range(F): # for each filter/kernel
            db[f] += dout[n, f].sum() # one bias/filter
            for i in range(0, Hout): # for each y activation
                for j in range(0, Wout): # for each x activation
                    dw[f] += x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] * dout[n, f, i, j]
                    dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += w[f] * dout[n, f, i, j]
    
    # extract dx from dx_pad since dx.shape needs to match x.shape
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    """
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = x.shape

    pool_height,pool_width,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_width) // stride

    out = np.zeros((N,C,Hout,Wout))
    
    for n in range(N):
      for c in range(C):
        for h in range(Hout):
          for w1 in range(Wout):
            out[n,c,h,w1] = np.amax(x[ n, c, h*stride:h*stride+pool_height, w1*stride:w1*stride+pool_width])
            #x_max = x[n,c,stride*h:stride*h+pool_height,stride*w1:stride*w1+pool_width]
            #out[n,c,h,w1] = np.amax(x_max)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x,pool_param =  cache

    N,C,H,W = x.shape

    pool_height,pool_width,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_width) // stride

    dx = np.zeros_like(x)

    for n in range(N):
      for c in range(C):
        for h in range(Hout):
          for w1 in range(Wout):
            index = np.argmax(x[ n, c, h*stride:h*stride+pool_height, w1*stride:w1*stride+pool_width])
            ind1,ind2 = np.unravel_index(index , (pool_height,pool_width))
            # ind1 and ind2 gives relative index to array index for exxample if index (5,5) has a max element in that part of x 
            # but argmax might give (1,2) relative to that part of x
            dx[n, c, h*stride:h*stride+pool_height, w1*stride:w1*stride+pool_width][ind1,ind2] = dout[n, c, h, w1]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x = x.transpose(0,2,3,1).reshape(N*H*W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0,3,1,2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout = dout.transpose(0,2,3,1).reshape(N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0,3,1,2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # copied from -https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment2/cs231n/layers.py
    # Note - Imlement from scratch later
    # concept - done 

    # key idea of Groupnorm: compute mean and variance statistics by dividing 
    # each datapoint into G groups 
    # gamma/beta (shift/scale) are per channel

    # using minimal-num-of-operations-per-step policy to ease the backward pass  

    N, C, H, W = x.shape
    size = (N*G, C//G * H * W) # in groupnorm, D = C//G * H * W

    # (0) rehsape X to accommodate G
    # divide each sample into G groups (G new samples)
    x = x.reshape((N*G, -1)) # reshape to same as size # reshape NxCxHxW ==> N*GxC/GxHxW =N1*C1 (N1>N*Groups)

    # (1) mini-batch mean by averaging over a particular column / feature dimension (D)
    # over each sample (N) in a minibatch 
    mean = x.mean(axis = 1, keepdims= True) # (N,1) # sum through D
    # can also do mean = 1./N * np.sum(x, axis = 1)

    # (2) subtract mean vector of every training example
    dev_from_mean = x - mean # (N,D)

    # (3) following the lower branch for the denominator
    dev_from_mean_sq = dev_from_mean ** 2 # (N,D)

    # (4) mini-batch variance
    var = 1./size[1] * np.sum(dev_from_mean_sq, axis = 1, keepdims= True) # (N,1)
    # can also do var = x.var(axis = 0)

    # (5) get std dev from variance, add eps for numerical stability
    stddev = np.sqrt(var + eps) # (N,1)

    # (6) invert the above expression to make it the denominator
    inverted_stddev = 1./stddev # (N,1)

    # (7) apply normalization
    # note that this is an element-wise multiplication using broad-casting
    x_norm = dev_from_mean * inverted_stddev # also called z or x_hat (N,D) 
    x_norm = x_norm.reshape(N, C, H, W)

    # (8) apply scaling parameter gamma to x
    scaled_x = gamma * x_norm # (N,D)

    # (9) shift x by beta
    out = scaled_x + beta # (N,D)

    # backprop sum axis
    axis = (0, 2, 3)

    # cache values for backward pass
    cache = {'mean': mean, 'stddev': stddev, 'var': var, 'gamma': gamma, \
             'beta': beta, 'eps': eps, 'x_norm': x_norm, 'dev_from_mean': dev_from_mean, \
             'inverted_stddev': inverted_stddev, 'x': x, 'axis': axis, 'size': size, 'G': G, 'scaled_x': scaled_x}

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

     # convention used is downstream gradient = local gradient * upstream gradient
    # extract all relevant params
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean, axis, size, G, scaled_x = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean'], cache['axis'], cache['size'], cache['G'], cache['scaled_x']

    N, C, H, W = dout.shape
    
    # (9)
    dbeta = np.sum(dout, axis = (0,2,3), keepdims = True) #1xCx1x1
    dscaled_x = dout # N1xC1xH1xW1

    # (8)
    dgamma = np.sum(dscaled_x * x_norm,axis = (0,2,3), keepdims = True) # N = sum_through_D,W,H([N1xC1xH1xW1]xN1xC1xH1xW1)
    dx_norm = dscaled_x * gamma # N1xC1xH1xW1 = [N1xC1xH1xW1] x[1xC1x1x1]
    dx_norm = dx_norm.reshape(size) #(N1*G,C1//G*H1*W1)

    # (7)
    dinverted_stddev = np.sum(dx_norm * dev_from_mean, axis = 1, keepdims = True) # N = sum_through_D([NxD].*[NxD]) =4×60
    ddev_from_mean = dx_norm * inverted_stddev #[NxD] = [NxD] x [Nx1]

    # (6)
    dstddev = (-1/(stddev**2)) * dinverted_stddev # N = N x [N]

    # (5)
    dvar = 0.5 * (1/np.sqrt(var + eps)) * dstddev # N = [N+const]xN

    # (4)
    ddev_from_mean_sq = (1/size[1]) * np.ones(size) * dvar # NxD = NxD*N

    # (3)    
    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq # [NxD] = [NxD]*[NxD]

    # (2)
    dx = (1) * ddev_from_mean # [NxD] = [NxD]
    dmean = -1 * np.sum(ddev_from_mean, axis = 1, keepdims = True) # N = sum_through_D[NxD]

    # (1) cache
    dx += (1/size[1]) * np.ones(size) * dmean # NxD (N= N1*Groups) += [NxD]XN

    # (0):
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

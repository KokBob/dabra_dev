# -*- coding: utf-8 -*-
"""
honu.py: Creates HONU neuron class. Can generate any desired order.

__doc__ using Sphnix Style
"""

# Copyright 2022 University Southern Bohemia

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__python__ = "3.8.0"

# Import libs for HONU
import numpy as np
from itertools import combinations_with_replacement
import math

# Scripts
def gen_x_y(data, u, y, pred):
    """
    Generates input matrix from given data for given recursion.

    Parameters
    ----------
    data : pandas.DataFrame
        Pandas with data to be processed. Propper column naming is expected.
    u : dict
        Dictionary of input columns which should be processed. With amount of
        historical values. May contain multiple keys.
    y : dict
        Dictionary with name of output value and number of historic values which
        should be placed in
    pred : int
        How far ahead should the target and inputs be shifted. Whats our prediction
        horizont.

    Returns
    -------
    X : numpy.Array
        Input matrix for neural networks. Shape is [N x M]
    Y : numpy.Array
        Output (target) matrix for neural netowrks. Shape is [N, ]

    Examples
    --------
    Following line will generate training matrix X from columns input and raw.
    From column input there will be 2 values time shifted and and from column
    raw there will be 3 values time shifted. Target vector Y will be generated
    from column pid. From target there will be placed 2 shifted values matrix X
    and the target shift will be 1 sample.
    Output shape of these function inputs will be X = [N, 7] ; Y = [N, ]

    >>> X, Y = gen_x_y(data, {"input":2, "raw":3}, {"pid":2}, 1)
    X, Y
    """
    ukeys = []
    nx = 0
    ny = 0
    offset = 0
    # Get intel from keys
    for key in u:
        nx += u[key]
        ukeys.append(key)
    for check, key in enumerate(y):
        ny = y[key]
        if check > 0:
            raise IOError("Y can only have one key.")
    # Prepare IO arrays
    X = np.zeros((data.shape[0], nx+ny))
    Y = np.zeros(data.shape[0])
    if pred == 0:
        Y[:] = data.loc[:, key].copy()
    elif pred > 0:
        Y[:-pred] = data.loc[:, key].iloc[pred:].copy()
    else:
        raise IOError("Prediction has to be positive number!")
    # Create X and Y
    for shift, uin in enumerate(u):
        ct = 0
        for colx in range(offset, u[ukeys[shift]]+offset):
            X[ct:-nx+ct, colx] = data.loc[:, uin].iloc[:-nx].values
            ct += 1
        offset = colx+1
    for coly in range(ny):
        scoly = coly+nx
        X[coly:-ny+coly, scoly] = data.loc[:, key].iloc[:-ny].values
    return X, Y


# Module class
class HONU:
    """
    Constructs HONU of any desired order given by parameter r. Autoupdates its state for given X and Y inputs.

    Parameters
    ----------
    nx : int
        Number of X inputs into HONU.
    ny : int
        Number of Y inputs into HONU. Commonly used previous Y to make prediction more precise. Can be 0.
    r : int
        Order of desired HONU. Careful with too high orders! Number of weights is defined as
        ((n+r-1)!/(r!)/(n-1)!; where n = nx+ny+1.
    w_div : numeric, optional
        Initial weights divider. For some applications it is more efficient to start with very small weights
    dtype : numpy.dtype, optional
        Type of numbers inside HONU. Sometimes lower precision of numbers is sufficient and can save a lot
        of computational performance.
    **wx
        dictionary containing w and x of saved HONU instance to reconstruct trained model.
    """

    def __init__(self, nx, ny, r, w_div=100, dtype=np.float64, **wx):
        """
        Docstring of class constructor. For full documentation do see class docstring.
        """
        self._dtype = dtype
        self._states = np.zeros(ny + nx + 1, dtype=self._dtype)
        self._states[0] = 1 #Bias
        self._r = r
        self._ny = ny
        self._nx = nx
        self._nw = self.get_nw_count(self._ny, self._nx, self._r)
        self._w = np.zeros(1)
        self._x = np.zeros(1)
        self._assign_wx(wx, w_div)
        self.__optim__ = None

    # Properties of HONU class
    @property
    def nw(self):
        return self._nw
    @property
    def order(self):
        return self._r
    @property
    def w(self):
        return self._w
    @property
    def state(self):
        return self._states
    @property
    def yn(self):
        return self._w.dot(self._x)
    @property
    def colx(self):
        return self._x

    # Hidden methods of HONU class
    def _assign_wx(self, wx, w_div):
        """
        Assigns given weights and last HONU state if given on init, else constructs new random weights
        and inits default colX.

        Parameters
        ----------
        wx : dict
            Class constructor keyword arguments containing w or x for reconstruction of HONU class.
        w_div : numeric
            Divider of new weights in case they are not provided in wx.
        """
        if 'w' in wx:
            if wx['w'].size != self.nw:
                raise ValueError("Weights must have size: %s" %(self.nw))
            else:
                self._w = np.copy(wx['w'])
        else:
            self._w = np.random.rand(self.nw).astype(self._dtype)/w_div
        if 'x' in wx:
            if wx['x'].size != self.nw:
                raise ValueError("Input must have size: %s" %(self.nw))
            else:
                self._x = np.copy(wx['x'])
        else:
            self._x = np.zeros(self.nw, dtype=self._dtype)
            self._x = self.gen_colx(self._states, self._r)

    def _updatestates(self, x, y=None):
        """
        Update the states vector one timestep. From k to k+1.

        Parameters
        ----------
        x : list
            List of values for updating state of HONU. Must match init size.
        y : numeric or None
            If Honu has been created with Y[k-1, ..], It is required parameter.
            Size must match.
            Defaults to None.
        """
        # Catch some of the common input errors
        if len(x) != self._nx:
            raise IndexError("Input X length (%s) must match the size of this HONU inputs (%s)" %(len(x), self._nx))
        if isinstance(y, type(None)) and self._ny > 0:
            raise ValueError("Y input can not be None. HONU expects Y[k-1]")
        if not isinstance(y, type(None)) and self._ny == 0:
            raise ValueError("HONU does not accept Y as input")

        # Update Y states
        if self._ny > 0:
            # Write more or single numbers
            try:
                # write all hist Y
                for pos, y_val in enumerate(y):
                    self._states[pos+1] = y[pos]
                ypos = pos+2
            except TypeError:
                self._states[1] = y
                ypos = 1
        else:
            ypos = 1

        # Update X states
        # Write more or single numbers
        try:
            # write all hist Y
            for pos, y_val in enumerate(x):
                self._states[pos+ypos] = x[pos]
        except TypeError:
            self._states[2+ypos] = x

    # Public methods of HONU class
    def gen_colx(self, x, r):
        """
        Generate colX for given input with itertools for any desired polynom

        Parameters
        ----------
        x : numpy.array
            input data to neuron
        r : int
            order of polynom

        Returns
        -------
        colx : numpy.array
            vector of input data with all expected permutations
        """
        colx = np.zeros(self.nw)
        for pos, com in enumerate(combinations_with_replacement(x, r)):
            colx[pos] = math.prod(com) # numpy was too slow here for due to Ctype copy
        return colx

    def update_weights(self, dw, learned=False):
        """
        Updates weights of HONU instance. If learned is true, returns how the weights would have changed.

        Parameters
        ----------
        dw : numpy.array
            weight changes for given update obtained by backpropagation. (HONUOptimizer)
        learned : bool, optional
            Specifies if HONU is considered trained. If False = weights are updated.
            If True = returns how the weights would have updated.
        """
        if learned:
            return self._w+dw
        else:
            self._w += dw

    def predict(self, x=None, y=None):
        """
        Predict output for desired inputs with active HONU instance

        Parameters
        ----------
        x : numpy.array or None, optional
            Input for HONU prediction. If None, current HONU state is used as input.
        y : numpy.array or None, optional
            Input of historic Y for HONU prediction, if HONU was trained with it as input.
            If None, current HONU state is used as input.

        Returns
        -------
        yn : numeric
            HONU output for given inputs.
        """
        if type(x) != type(None):
            self._updatestates(x, y)
        self._x = self.gen_colx(self._states, self._r)
        return self._w.dot(self._x)

    def fit(self, x, target, y=None, optimizer="gd", return_elements=False, e=None, **kwargs):
        """
        Fit HONU on input data for given target.
        At this moment for HONU only GD is supported. in this class as example
        for more advanced optimizer implementation later on.

        Parameters
        ----------
        x : numpy.Array
            X matrix for given instance of GHONU. Size has to match GHONU init size.
            Can not be none!
        target : numeric
            Target of supervised learning for given inputs. Used in adaptation methods.
        y : numpy.Array
            Part of Y matrix for prediction, if GHONU has been instanced with it,
            it can not remain none and has to match init size. Can be None.
        optimizer : str, optional
            Selects backend optimizer. Currently supported are: ["gd", "gngd", "nlms", "adam"].
            Defaults to "gd" - Gradient Descent
        mu : numeric, optional
            Learning rate of predictor neuron. Defaults to 0.001.
        return_elements : Bool, optional
            Toggles whether training parameters should be returned to caller.
            If true, returns yn, yp, yg, e, dw_pred, dw_gate, predictor.w, gate.w
        e : numeric, optional
            If not none, custom error is applied to training. Should be used if
            HONU instance is part of network!

        Returns
        -------
        if return elements
            Returns parameters of update such as yn, e, wall, dw
        else
            None
        """

        # Get output of current HONU state
        yn = self.predict(x, y)

        # Calculate error if not part of the network
        if e == None:
            e = target - yn

        # weight change update by selected optimizer
        if optimizer.lower() == "gd":
            # weight update
            dw = gradient_descent(e, self.colx, **kwargs)

        elif optimizer.lower() == "gngd":
            # If GNGD isnt init start its class
            if self.__optim__ == None:
                self.__optim__ = Generalized_Normalized_Gradient_Descent(self.colx, **kwargs)
            # weight update
            dw = self.__optim__.update(e, self.colx)

        elif optimizer.lower() == "nlms":
            # weight update
            dw = normalized_least_mean_squares(e, self.colx, **kwargs)

        elif optimizer.lower() == "adam":
            # If ADAM isnt init, start its class
            if self.__optim__ == None:
                self.__optim__ = Adam(self.colx, **kwargs)
            # weight update
            dw = self.__optim__.update(e, self.colx)

        else:
            raise IOError("Unknown optimizer method. See docstring for supported methods.")

        # Update HONU weights
        self.update_weights(dw)

        # If return elements
        if return_elements:
            return [yn, e, self.w, dw]

    @staticmethod
    def get_nw_count(ny, nx, r):
        """
        Combinations with repetition = ((n+r-1)!/(r!)/(n-1)!; n = nx + ny + 1 (bias)
        where r is the neuron polynomial order and n is the number of the states

        Parameters
        ----------
        nx : int
            length of input data
        r : int
            order of polynom

        Returns
        -------
        nw_count : int
            number of required weights for all expected combinations
        """
        n = (ny + nx + 1)
        nw_count = int(np.ceil(math.factorial(n+r-1)/math.factorial(r)/math.factorial(n-1)))
        return nw_count

# Optimizer methods
def gradient_descent(e, colx, mu=0.01, **kwargs):
    """
    Gradient descent weight update method

    Parameters
    ----------
    e : numeric
        Error of current update.
    colx : np.Array
        Col X with current values used for back propagation.
    mu : numeric, optional
        Learning rate of gradient descent method. The default is 0.01.

    Returns
    -------
    dw : np.array
        Weight updates for given data.

    """
    # GD algorithm
    dw = mu * e * colx
    return dw

class Generalized_Normalized_Gradient_Descent():
    """
    The generalized normalized gradient descent (GNGD)method is an extension of
    the NLMS method (Normalized Least-mean-squares (NLMS)).
    """

    def __init__(self, colx, mu=1., eps=1., ro=0.1):
        """
        The generalized normalized gradient descent (GNGD)method is an extension of
        the NLMS method (Normalized Least-mean-squares (NLMS)).

        Parameters
        ----------
        colx : np.Array
            Colx vector of associated honu (for size estimations).
        mu : numeric, optional
            Learning rate. The default is 1..
        eps : numeric, optional
            epsilon for normalized gd. The default is 1..
        ro : numeric, optional
            ro for generalited normalized gd. The default is 0.1.

        Returns
        -------
        None.

        """
        self.mu = mu
        self.eps = eps
        self.ro = ro
        self.last_e = 0
        self.last_x = np.zeros(colx.shape)

    def update(self, e, colx):
        """
        Calculates weight update for given error and data

        Parameters
        ----------
        e : numeric
            Error for given sample.
        colx : np.Array
            Colx vector of associated HONU.

        Returns
        -------
        dw : np.array
            Weight updates for associated HONU.

        """
        # GNGD algorithm
        self.eps = self.eps - self.ro * self.mu * e * self.last_e * \
                   np.dot(colx, self.last_x) / \
                   (np.dot(self.last_x, self.last_x) + self.eps) ** 2
        nu = self.mu / (self.eps + np.dot(colx, colx))
        self.last_e, self.last_x = e, colx
        dw = nu * e * colx
        return dw

def normalized_least_mean_squares(e, colx, mu=0.1, eps=0.001):
    """
    Normalized Least Mean Squares for weight update rule method.

    Parameters
    ----------
    e : numeric
        Error for given sample.
    colx : np.Array
        Colx vector of associated HONU.
    mu : numeric, optional
        Learning rate for given update. The default is 0.1.
    eps : numeric, optional
        regularization term (float). It is introduced to preserve
        stability for close-to-zero input vectors. The default is 0.001.

    Returns
    -------
    dw : np.Array
        Weight updates for associated HONU.

    """

    dw = mu / (eps + np.dot(colx, colx)) * colx * e
    return dw

class Adam():

    def __init__(self, colx, mu=0.01, beta_1=0.9, beta_2=0.999, eta=1e-8):
        """
        Adam optimizer for HONU weight updates

        Parameters
        ----------
        colx : np.array
            ColX vector of associated HONU.
        mu : numeric, optional
            Learning rate for given update. The default is 0.01.
        beta_1 : numeric, optional
            [0, 1) exponential decay rate for the moment estimates. The default is 0.9.
        beta_2 : numeric, optional
            [0, 1) exponential decay rate for the moment estimates.. The default is 0.999.
        eta : numeric, optional
            Something very small to prevent division by zero. The default is 1e-8.
        """
        self.size = colx.shape # Size of the X vector of HONU
        self.mu = mu # learning rate
        self.beta_1 = beta_1 # [0, 1) exponential decay rate for the moment estimates
        self.beta_2 = beta_2 # beta_1 and beta_2
        self.eta = eta # Very small value to prevent zero-division
        self.moment_1 = np.zeros(self.size) # Initial moment m1<--0
        self.moment_2 = np.zeros(self.size) # Initial moment m2<--0
        self._k = 0 # Initial step

    def update(self, e, colx):
        """
        Calculates weight update for given error and data

        Parameters
        ----------
        e : numeric
            Error for given sample.
        colx : np.Array
            Colx vector of associated HONU.

        Returns
        -------
        dw : np.array
            Weight updates for associated HONU.

        """

        gradient = colx * e

        self.moment_1 = self.moment_1 * self.beta_1 + (1 - self.beta_1) * gradient
        self.moment_2 = self.moment_2 * self.beta_2 + (1 - self.beta_2) * gradient**2

        moment_1_hat = self.moment_1 / (1 - self.beta_1**(self._k+1))
        moment_2_hat = self.moment_2 / (1 - self.beta_2**(self._k+1))

        self._k += 1

        dw = self.mu * moment_1_hat / (np.sqrt(moment_2_hat) + self.eta * np.ones(self.size))
        return dw


if __name__ == "__main__":
    # Create QNU
    qnu = HONU(2, 0, 2)
    print("Number of weights: %s" %(qnu.nw))

    # Create data
    N = np.arange(0, 60, 0.1)
    temp = np.sin(N)+np.random.randn(N.shape[0])/10
    Y = temp[3:]
    X = np.zeros((N.shape[0]-2,2))
    X[:,0] = temp[1:-1]
    X[:,1] = temp[2:]

    # Training setups
    epochs = 10
    mu = 0.1
    eo = np.zeros(N.shape[0]-3)
    wall = np.zeros((N.shape[0]-3, qnu.nw))
    yn = np.copy(eo)

    # Epoch magic
    for epochs in range(epochs):
        # One epoch online
        for k in range(0,N.shape[0]-3):
            yn[k], eo[k], wall[k, :], dw = qnu.fit(X[k], Y[k], return_elements=True)

    # Get MSE for given settings
    print("MSE:", np.square(eo).mean())
    # Get some random for comparison
    nran = np.random.randint(0, high=yn.shape[0])
    print("\nIO comparison for index %s\nInput: %s\nReal output: %s\nNeuron output:%s" %(nran, X[nran], Y[nran], yn[nran]))

    # Plot results for complete script overview
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(Y, "b", label="$Y_{original}$")
    ax[0].plot(yn, "g", label="$Y_{neuron}$")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("$[/]$")

    ax[1].plot(eo, "r", label="$Error$")
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel("$[/]$")
    ax[1].set_xlabel("$Sample~[k]$")

    fig.suptitle("HONU sample prediction on test data from module script")
    plt.tight_layout()
    plt.show()

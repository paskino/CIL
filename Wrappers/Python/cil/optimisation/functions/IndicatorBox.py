# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author(s): 
# Evangelos Papoutsellis (UKRI)
# Edoardo Pasca (UKRI)
# Gemma Fardell (UKRI)

from cil.optimisation.functions import Function
import numpy as np
import numba
from cil.utilities import multiprocessing as cil_mp


class IndicatorBox(Function):
    
    
    r'''Indicator function for box constraint
            
      .. math:: 
         
         f(x) = \mathbb{I}_{[a, b]} = \begin{cases}  
                                            0, \text{ if } x \in [a, b] \\
                                            \infty, \text{otherwise}
                                     \end{cases}
    
    '''
    
    def __init__(self, lower=None, upper=None):
        '''creator

        Parameters
        ----------
        
            lower : float, DataContainer or numpy array, default None
                Lower bound. If set to None, it is equivalent to ``-np.inf``.
            upper : float, DataContainer or numpy array, default None
                Upper bound. If set to None, it is equivalent to ``np.inf``.
        
        If passed a ``DataContainer`` or ``numpy array``, the bounds can be set to different values for each element.

        To suppress the evaluation of the function, set ``suppress_evaluation`` to ``True``. This will return 0 for any input.

        Example:
        --------

        .. code-block:: python

          ib = IndicatorBox(lower=0, upper=1)
          ib.set_suppress_evaluation(True)
          ib.evaluate(x) # returns 0
        '''
        super(IndicatorBox, self).__init__()
        
        # We set lower and upper to either a float or a numpy array        
        self.lower = -np.inf if lower is None else _get_as_nparray_or_number(lower)
        self.upper =  np.inf if upper is None else _get_as_nparray_or_number(upper)

        self.orig_lower = lower
        self.orig_upper = upper
        # default is to evaluate the function
        self._suppress_evaluation = False

    @property
    def suppress_evaluation(self):
        return self._suppress_evaluation

    def set_suppress_evaluation(self, value):
        '''Suppresses the evaluation of the function
        
        Parameters
        ----------

            value : bool
                If True, the function evaluation on any input will return 0, without calculation.
        '''
        if not isinstance(value, bool):
            raise ValueError('Value must be boolean')
        self._suppress_evaluation = value

    def __call__(self,x):
        '''Evaluates IndicatorBox at x
        
        Parameters
        ----------
        
            x : DataContainer
            
        Evaluates the IndicatorBox at x. If ``suppress_evaluation`` is ``True``, returns 0.  
        '''
        if not self.suppress_evaluation:
            return self.evaluate(x)    
        return 0.0

    def evaluate(self,x):
        
        '''Evaluates IndicatorBox at x'''
        
        num_threads = numba.get_num_threads()
        numba.set_num_threads(cil_mp.NUM_THREADS)
        breaking = np.zeros(numba.get_num_threads(), dtype=np.uint8)
                
        if isinstance(self.lower, np.ndarray):
            if isinstance(self.upper, np.ndarray):

                _array_within_limits_aa(x.as_array(), self.lower, self.upper, breaking)

            else:

                _array_within_limits_af(x.as_array(), self.lower, self.upper, breaking)

        else:
            if isinstance(self.upper, np.ndarray):

                _array_within_limits_fa(x.as_array(), self.lower, self.upper, breaking)

            else:

                _array_within_limits_ff(x.as_array(), self.lower, self.upper, breaking)

        numba.set_num_threads(num_threads)
        return np.inf if breaking.sum() > 0 else 0.0
    
    def gradient(self,x):
        '''IndicatorBox is not differentiable, so calling gradient will raise a ``ValueError``'''
        return ValueError('Not Differentiable') 
    
    def convex_conjugate(self,x):
        '''Convex conjugate of IndicatorBox at x'''
        # set the number of threads to the number of threads used in the CIL multiprocessing module
        num_threads = numba.get_num_threads()
        numba.set_num_threads(cil_mp.NUM_THREADS)

        acc = np.zeros((numba.get_num_threads()), dtype=np.uint32)
        _convex_conjugate(x.as_array(), acc)
        
        # reset the number of threads to the original value
        numba.set_num_threads(num_threads)
        
        return np.sum(acc)
         
    def proximal(self, x, tau, out=None):
        
        r'''Proximal operator of IndicatorBox at x

        .. math:: prox_{\tau * f}(x)

        Parameters
        ----------

        x : DataContainer
            Input to the proximal operator
        tau : float
            Step size. Notice it is ignored in IndicatorBox
        out : DataContainer, optional
            Output of the proximal operator. If not provided, a new DataContainer is created.

        Note
        ----

            ``tau`` is ignored but it is in the signature of the generic Function class
        '''
        should_return = False
        if out is None:
            should_return = True
            out = x.copy()
        else:
            out.fill(x)
        outarr = out.as_array()

        # the following could be achieved by the following, but it is 2x slower
        # np.clip(outarr, None if self.orig_lower is None else self.lower, 
        #                 None if self.orig_upper is None else self.upper, out=outarr)
        if self.orig_lower is not None and self.orig_upper is not None:
            if isinstance(self.lower, np.ndarray):
                if isinstance(self.upper, np.ndarray):
                    _proximal_aa(outarr, self.lower, self.upper)
                else:
                    _proximal_af(outarr, self.lower, self.upper)
            
            else:
                if isinstance(self.upper, np.ndarray):
                    _proximal_fa(outarr, self.lower, self.upper)
                else:
                    np.clip(outarr, self.lower, self.upper, out=outarr)

        elif self.orig_lower is None:
            if isinstance(self.upper, np.ndarray):
                _proximal_na(outarr, self.upper)
            else:
                np.clip(outarr, None, self.upper, out=outarr)
        
        elif self.orig_upper is None:
            if isinstance(self.lower, np.ndarray):
                _proximal_an(outarr, self.lower)
            else:
                np.clip(outarr, self.lower, None,out=outarr)

        out.fill(outarr)
        if should_return:
            return out
            
    def proximal_conjugate(self, x, tau, out=None):
        
        r'''Proximal operator of the convex conjugate of IndicatorBox at x:

          .. math:: prox_{\tau * f^{*}}(x)

          Parameters
          ----------

            x : DataContainer
                Input to the proximal operator
            tau : float
                Step size. Notice it is ignored in IndicatorBox, see ``proximal`` for details
            out : DataContainer, optional
                Output of the proximal operator. If not provided, a new DataContainer is created.

        '''

        # x - tau * self.proximal(x/tau, tau)
        should_return = False
        
        if out is None:
            out = self.proximal(x, tau)
            should_return = True
        else:
            self.proximal(x, tau, out=out)
        
        out.sapyb(-1., x, 1., out=out)

        if should_return:
            return out
        
## Utilities
def _get_as_nparray_or_number(x):
    '''Returns x as a numpy array or a number'''
    try:
        return x.as_array()
    except AttributeError:
        # In this case we trust that it will be either a numpy ndarray 
        # or a number as described in the docstring
        return x

@numba.jit(nopython=True, parallel=True)
def _array_within_limits_ff(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    arr = x.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < lower or arr[i] > upper):
            breaking[j] = 1

@numba.jit(nopython=True, parallel=True)
def _array_within_limits_af(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != lower.size:
        raise ValueError('x and lower must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < loarr[i] or arr[i] > upper):
            breaking[j] = 1

@numba.jit(parallel=True, nopython=True)
def _array_within_limits_aa(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != lower.size or x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < loarr[i] or arr[i] > uparr[i]):
            breaking[j] = 1

@numba.jit(nopython=True, parallel=True)
def _array_within_limits_fa(x, lower, upper, breaking):
    '''Returns 0 if all elements of x are within [lower, upper]'''
    if x.size != upper.size:
        raise ValueError('x and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
        
        if breaking[j] == 0 and (arr[i] < lower or arr[i] > uparr[i]):
            breaking[j] = 1

##########################################################################

@numba.jit(nopython=True, parallel=True)
def _proximal_aa(x, lower, upper):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != lower.size or x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        if arr[i] < loarr[i]:
            arr[i] = loarr[i]
        if arr[i] > uparr[i]:
            arr[i] = uparr[i]

@numba.jit(nopython=True, parallel=True)
def _proximal_af(x, lower, upper):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != lower.size :
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        if arr[i] < loarr[i]:
            arr[i] = loarr[i]
        if arr[i] > upper:
            arr[i] = upper
    

@numba.jit(nopython=True, parallel=True)
def _proximal_fa(x, lower, upper):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != upper.size:
        raise ValueError('x, lower and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        if arr[i] < lower:
            arr[i] = lower
        if arr[i] > uparr[i]:
            arr[i] = uparr[i]

@numba.jit(nopython=True, parallel=True)
def _proximal_na(x, upper):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != upper.size:
        raise ValueError('x and upper must have the same size')
    arr = x.ravel()
    uparr = upper.ravel()
    for i in numba.prange(x.size):
        if arr[i] > uparr[i]:
            arr[i] = uparr[i]

@numba.jit(nopython=True, parallel=True)
def _proximal_an(x, lower):
    '''Similar to np.clip except that the clipping range can be defined by ndarrays'''
    if x.size != lower.size:
        raise ValueError('x and lower must have the same size')
    arr = x.ravel()
    loarr = lower.ravel()
    for i in numba.prange(x.size):
        if arr[i] < loarr[i]:
            arr[i] = loarr[i]

@numba.jit(nopython=True, parallel=True)
def _convex_conjugate(x, acc):
    '''Convex conjugate of IndicatorBox
    
    im.maximum(0).sum()
    '''
    arr = x.ravel()
    j = 0
    for i in numba.prange(x.size):
        j = numba.np.ufunc.parallel._get_thread_id()
    
        if arr[i] > 0:
            acc[j] += arr[i]
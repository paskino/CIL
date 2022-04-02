# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.optimisation.functions import Function
from cil.framework import BlockDataContainer
import numpy as np
from numbers import Number
has_numba = True
try:
    import numba
except ImportError:
    has_numba = False
    

@numba.jit(nopython=True)
def _proximal_step_numba(arr, abstau):
    '''Numba implementation of a step in the calculation of the proximal of MixedL21Norm
    
    Parameters:
    -----------
    arr : numpy array, best if contiguous memory. 
    abstau: float >= 0

    Returns:
    --------
    Stores the output in the input array. Returns the modified input numpy array. 

    Note:
    -----
    
    Input arr should be contiguous for best performance'''
    tmp = arr.ravel()
    for i in numba.prange(tmp.size):
        if tmp[i] == 0:
            continue
        a = tmp[i] / abstau
        el = a - 1
        if el <= 0.0:
            el = 0.
        
        tmp[i] = el / a 
    return arr
def _proximal_step_numpy(tmp, tau):
    '''Numpy implementation of a step in the calculation of the proximal of MixedL21Norm
    
    Parameters:
    -----------
    tmp : DataContainer/ numpy array, best if contiguous memory. 
    tau: float or DataContainer

    Returns:
    --------

    A DataContainer where we have substituted nan with 0.
    '''
    # Note: we divide x by tau so the cases of tau both scalar and 
    # DataContainers run
    tmp /= np.abs(tau, dtype=np.float32)
    res = tmp - 1
    res.maximum(0.0, out=res)
    res /= tmp

    resarray = res.as_array()
    resarray[np.isnan(resarray)] = 0
    res.fill(resarray)
    return res

class MixedL21Norm(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()  
                    
        
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 

        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
              
        return x.pnorm(p=2).sum()                                
            
                            
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
        
        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
        
        i.e., 
        
        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
        
        where, 
        
        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
        
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                                        
        tmp = (x.pnorm(2).max() - 1)
        if tmp<=1e-5:
            return 0
        else:
            return np.inf
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """
                
        tmp = x.pnorm(2)
        if has_numba and isinstance(tau, Number):
            resarray = _proximal_step_numba(np.asarray(tmp.as_array(), order='C', dtype=np.float32), 
                                np.abs(tau, dtype=np.float32))
            res = tmp
            # may involve a copy if the data is not contiguous
            res.fill(resarray)
        else:
            res = _proximal_step_numpy(tmp, tau)
        
        if out is None:
            res = x.multiply(res)
        else:
            x.multiply(res, out = out)
            res = out

        if out is None:
            return res

class SmoothMixedL21Norm(Function):
    
    """ SmoothMixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \epsilon^2 + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
        
        Conjugate, proximal and proximal conjugate methods no closed-form solution
        
    
    """    
        
    def __init__(self, epsilon):
                
        r'''
        :param epsilon: smoothing parameter making MixedL21Norm differentiable 
        '''

        super(SmoothMixedL21Norm, self).__init__(L=1)
        self.epsilon = epsilon   
                
        if self.epsilon==0:
            raise ValueError('We need epsilon>0. Otherwise, call "MixedL21Norm" ')
                            
    def __call__(self, x):
        
        r"""Returns the value of the SmoothMixedL21Norm function at x.
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
            
            
        return (x.pnorm(2).power(2) + self.epsilon**2).sqrt().sum()
         

    def gradient(self, x, out=None): 
        
        r"""Returns the value of the gradient of the SmoothMixedL21Norm function at x.
        
        \frac{x}{|x|}
                
                
        """     
        
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                   
        denom = (x.pnorm(2).power(2) + self.epsilon**2).sqrt()
                          
        if out is None:
            return x.divide(denom)
        else:
            x.divide(denom, out=out)        

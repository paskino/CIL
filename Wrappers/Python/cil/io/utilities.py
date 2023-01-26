# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Author(s): Edoardo Pasca

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

def get_compress(compression=0):
    '''Returns whether the data needs to be compressed and to which numpy type
    
    Parameters:
    -----------
    compression : int, specifies the number of bits to use for compression, allowed values are 0, 8, 16. Default is 0, no compression.
    
    Returns:
    --------
    compress : bool, True if compression is required, False otherwise
    
    '''
    if compression == 0:
        compress = False
    elif compression == 8:
        compress = True
    elif compression == 16:
        compress = True
    else:
        raise ValueError('Compression bits not valid. Got {0} expected value in {1}'.format(compression, [0,8,16]))

    return compress

def get_compressed_dtype(data, compression=0):
    '''Returns whether the data needs to be compressed and to which numpy type
    
    Given the data and the compression level, returns the numpy type to be used for compression.

    Parameters:
    -----------
    data : DataContainer, numpy array, the data to be compressed
    compression : int, specifies the number of bits to use for compression, allowed values are 0, 8, 16. Default is 0, no compression.

    Returns:
    --------
    dtype : numpy type, the numpy type to be used for compression
    '''
    if compression == 0:
        dtype = data.dtype
    elif compression == 8:
        dtype = np.uint8
    elif compression == 16:
        dtype = np.uint16
    else:
        raise ValueError('Compression bits not valid. Got {0} expected value in {1}'.format(compression, [0,8,16]))

    return dtype

def get_compression_scale_offset(data, compression=0):
    '''Returns the scale and offset to be applied to the data to compress it
    
    Parameters:
    -----------
    data : DataContainer, numpy array, the data to be compressed
    compression : int, specifies the number of bits to use for compression, allowed values are 0, 8, 16. Default is 0, no compression.

    Returns:
    --------
    scale : float, the scale to be applied to the data for compression to the specified number of bits
    offset : float, the offset to be applied to the data for compression to the specified number of bits
    '''

    if compression == 0:
        # no compression
        # return scale 1.0 and offset 0.0
        return 1.0, 0.0

    dtype = get_compressed_dtype(data, compression)
    save_range = np.iinfo(dtype).max

    data_min = data.min()
    data_range = data.max() - data_min

    if data_range > 0:
        scale = save_range / data_range
        offset = - data_min * scale
    else:
        scale = 1.0
        offset = 0.0
    return scale, offset

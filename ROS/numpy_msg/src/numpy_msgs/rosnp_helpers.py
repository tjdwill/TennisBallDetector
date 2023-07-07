"""
@author: Terrance Williams
@date: 6 July 2023
@description: Helper functions to make ROSNumpy operations easier for the user.
"""
import numpy as np
from numpy_msg.msg import ROSNumpy


def construct_rosnumpy(array):
    """
    Construct a ROSNumpy message from a provided ndarray
    Because Numpy arrays are contiguous in memory, 
    we can flatten the array and reconstruct it if
    we know the shape and datatype.

    Parameter(s):
    array: np.ndarray

    Output(s):
    msg: ROSNumpy
    """
    # Get array information
    shape = array.shape
    dtype = array.dtype.name
    data = array.reshape(-1)  # flatten 
    # Generate message
    msg = ROSNumpy()
    msg.shape, msg.dtype, msg.rosnp = shape, dtype, data
    return msg


def open_rosnumpy(msg):
    """
    Reconstructs the original array from a ROSNumpy msg.

    Parameter(s):
    msg: ROSNumpy

    Outputs(s):
    result_array: np.ndarray
    """
    # Unpack array
    shape, dtype, data = msg.shape, msg.dtype, msg.rosnp
    result_array = np.array(data, dtype=dtype).reshape(shape)
    return result_array
    

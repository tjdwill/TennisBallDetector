"""
@author: Terrance Williams
@date: 6 July 2023
@description: Helper functions to make ROSNumpy operations easier for the user.
"""
import numpy as np
from numpy_msg.msg import ROSNumpy


def construct_rosnumpy(array: np.ndarray) -> ROSNumpy:
    """
    Construct a ROSNumpy message from a provided ndarray
    Because Numpy arrays are contiguous in memory, 
    we can flatten the array and reconstruct it if
    we know the shape and datatype.
    """
    # Get array information
    shape = array.shape
    dtype = array.dtype.name
    data = array.reshape(-1)  # flatten 
    # Generate message
    msg = ROSNumpy()
    msg.shape, msg.dtype, msg.data = shape, dtype, data
    return msg


def open_rosnumpy(msg: ROSNumpy) -> np.ndarray:
    """
    Reconstructs the original array from a ROSNumpy msg.
    """
    # Unpack array
    shape, dtype, data = msg.shape, msg.dtype, msg.data
    result_array = np.array(data, dtype=dtype).reshape(shape)
    return result_array
    
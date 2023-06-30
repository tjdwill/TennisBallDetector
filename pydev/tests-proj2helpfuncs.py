# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:43:35 2023

@author: Tj
"""

import sys
def add_module(modules):
    for module in modules:
        if module not in sys.path:
            sys.path.append(module)
            
proj2help = r'C:\Users\Tj\Documents\Graduate_School\UWF\Engineering_M.S\Summer_2023\CAP6665-Computer_Vision\Projects\Project_2\Code\scripts\tools'.replace('\\','/')
add_module([proj2help])

import numpy as np
import proj2helpfuncs as phf
from numpy.random import randint

def test_get_ground_pixel() -> bool:
    test1 = (100,100)
    test2 = (480, 640)
    test3 = (1080, 1920)
    
    assert phf.get_ground_pixel(test1) == (99, 50)
    assert phf.get_ground_pixel(test2) == (479, 320)
    assert phf.get_ground_pixel(test3) == (1079, 960)
    print("PASS: get_ground_pixel\n")
    return True
    
    
def test_calc_pixel_angle() -> bool:
    ''' Test cases for the phf.calc_pixel_angles function'''
    ground_pixel = (4,2)
    assert phf.calc_pixel_angle((4,4), ground_pixel) == -90
    assert phf.calc_pixel_angle((4,0), ground_pixel) == 90
    assert phf.calc_pixel_angle((3,2), ground_pixel) == 0
    assert phf.calc_pixel_angle((0,2), ground_pixel) == 0
    assert phf.calc_pixel_angle((2,0), ground_pixel) == 45
    assert phf.calc_pixel_angle((2,4), ground_pixel) == -45
    try:
        assert phf.calc_pixel_angle((4,2), ground_pixel) 
    except ValueError:
        print("calc_pixel_angle: Same-point-check passed.")
    print("PASS: calc_pixel_angle\n")
    
    return True

def test_process_centers() -> bool:
    """
    Ensure process_centers function:
        - flattens data to one dimension, 
        - removes radius data (third col)
        - chooses entry with the most points as the initial mean
        - the number of segments equals length of this mean.
    """
    # Create test data
    SEGMENTS = 10
    
    data = [randint(0,10,size=(1,1,3)),
            randint(0,10,size=(1,SEGMENTS,3)),
            randint(0,10,size=(1,SEGMENTS-1,3))]
    flat_data, means, k = phf.process_centers(data)
    
    # tests
    try:
        assert flat_data[0].ndim == data[0].ndim - 2
        assert flat_data[0].shape[-1] == 2
        assert all(np.equal(means[0], data[1][0, 0, 0:-1]))
        assert k == SEGMENTS
    except AssertionError:
        print("FAIL: Process Centers TEST")
        return data, flat_data, means, k
    else:
        print("PASS: Process Centers Test.\n")
    return True
    
if __name__ == '__main__':
    test_calc_pixel_angle()
    test_get_ground_pixel()
    test_process_centers()
    
    

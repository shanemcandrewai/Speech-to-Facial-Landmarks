""" test utilities """
import numpy as np
def get_array(shuffle=True, shape=None):
    """ create a simple integer ndarray with missing and unique elements"""
    if shape is None:
        shape = (3, 3, 2)
    lmarks = np.arange(np.prod(shape))
    if shuffle:
        np.random.shuffle(lmarks)
    lmarks.resize(shape)
    return np.concatenate((np.full((1, shape[1], shape[2]), np.nan), lmarks))

def get_fixed_array():
    """ create a fixed array for testing shape (4, 3, 2) """
    return np.array([[[np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan]],

                     [[3., 7.],
                      [2., 4.],
                      [1., 16.]],

                     [[17., 5.],
                      [10., 0.],
                      [14., 12.]],

                     [[8., 6.],
                      [13., 11.],
                      [9., 15.]]])

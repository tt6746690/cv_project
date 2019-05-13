import unittest 
import numpy as np
import demosaicing as dm

class TestBayer(unittest.TestCase):

    def test_bayer_downsample(self):
        l = np.arange(0,32).reshape((4,8))[:,:,np.newaxis]
        l = np.tile(l, (1,1,3))
        d = dm.bayer_downsample(l)

        np.testing.assert_equal(
            np.array([ [ 0,  0,  0,  0,  0,  0,  0,  0],
                       [ 0,  9,  0, 11,  0, 13,  0, 15],
                       [ 0,  0,  0,  0,  0,  0,  0,  0],
                       [ 0, 25,  0, 27,  0, 29,  0, 31]])
            , d[:,:,0])

        np.testing.assert_equal(
            np.array([[ 0,  1,  0,  3,  0,  5,  0,  7],
                        [ 8,  0, 10,  0, 12,  0, 14,  0],
                        [ 0, 17,  0, 19,  0, 21,  0, 23],
                        [24,  0, 26,  0, 28,  0, 30,  0]])
            , d[:,:,1])

        np.testing.assert_equal(
            np.array([  [ 0,  0,  2,  0,  4,  0,  6,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [16,  0, 18,  0, 20,  0, 22,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0]])
            , d[:,:,2])
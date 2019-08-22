from unittest import TestCase
from RockPy.tools.compute import *
import numpy as np
from numpy import testing


class TestCompute(TestCase):
    xyz = np.array([[0.6646505, 0.06948398, 0.35858963],
                    [0.66833224, 0.81348924, 0.8011098],
                    [0.48691201, 0.46500924, 0.10241104],
                    [0.5350718, 0.9985017, 0.77377981],
                    [0.60857609, 0.72430276, 0.17601995],
                    [0.12231943, 0.15982662, 0.91712825],
                    [0.25337639, 0.00669908, 0.71246859],
                    [0.23751526, 0.25943859, 0.76675908],
                    [0.6436179, 0.23422729, 0.56300565],
                    [0.15436119, 0.70173326, 0.85098924],
                    [0.44403356, 0.88315021, 0.58146584],
                    [0.83782532, 0.61483058, 0.69972283],
                    [0.3877226, 0.83629995, 0.6932362],
                    [0.60326719, 0.04427073, 0.75908668],
                    [0.70158067, 0.80292771, 0.425227],
                    [0.00866282, 0.22109461, 0.12163454],
                    [0.94286931, 0.69872257, 0.90944086],
                    [0.22941293, 0.12411983, 0.39646348],
                    [0.89913404, 0.95830739, 0.31119578],
                    [0.30845017, 0.49037893, 0.39102733],
                    [0.6493535, 0.85968252, 0.0983001],
                    [0.79230792, 0.90251372, 0.13450918],
                    [0.44342237, 0.3721808, 0.71984783],
                    [0.24780637, 0.83618377, 0.89200627],
                    [0.0419931, 0.00139568, 0.1307576],
                    [0.62182711, 0.69635268, 0.30799005],
                    [0.17518174, 0.0670227, 0.74722737],
                    [0.57023452, 0.86860249, 0.08756121],
                    [0.39825897, 0.9985351, 0.86477404],
                    [0.51391218, 0.07038537, 0.94681043]])

    dim = np.array([[5.96814386, 28.21764312, 0.75840282],
                    [50.5947583, 37.26812874, 1.32295489],
                    [43.68191314, 8.64872842, 0.68103225],
                    [61.8142688, 34.33500101, 1.3718756],
                    [49.96223868, 10.53998694, 0.96226938],
                    [52.57223895, 77.62272034, 0.93895198],
                    [1.51450422, 70.41668212, 0.75621159],
                    [47.52598981, 65.35729691, 0.84358839],
                    [19.99760495, 39.42054415, 0.88661253],
                    [77.59414898, 49.82481092, 1.11375026],
                    [63.307494, 30.46549618, 1.14683156],
                    [36.27274863, 33.95311614, 1.25282878],
                    [65.1267747, 36.94467613, 1.15338755],
                    [4.19712413, 51.44989418, 0.97062031],
                    [48.85374952, 21.74224765, 1.14792262],
                    [87.75621276, 28.79870465, 0.25249323],
                    [36.54065861, 37.77388087, 1.484688],
                    [28.41482448, 56.65869512, 0.47457277],
                    [46.82468039, 13.32318175, 1.35042137],
                    [57.82995454, 34.01836814, 0.69893875],
                    [52.93475126, 5.21330511, 1.0818396],
                    [48.72042111, 6.39062201, 1.20846],
                    [40.00797988, 51.19313885, 0.92375475],
                    [73.49260152, 45.64550551, 1.24751212],
                    [1.90357817, 72.18625101, 0.13734234],
                    [48.23587102, 18.25779106, 0.98307369],
                    [20.9363275, 75.90904477, 0.77040861],
                    [56.71524003, 4.81693162, 1.04273902],
                    [68.25574927, 38.81392735, 1.37967992],
                    [7.79870406, 61.28388356, 1.0795878]])

    def test_convert_to_xyz(self):
        true = self.xyz
        testing.assert_array_almost_equal(true, convert_to_xyz(self.dim))
        testing.assert_array_almost_equal(true.T, convert_to_xyz(self.dim.T))
        self.assertEqual(np.shape([1,2,3]), convert_to_xyz([1,2,3]).shape)
        self.assertEqual(self.dim.shape, convert_to_xyz(self.dim).shape)
        self.assertEqual(self.dim.T.shape, convert_to_xyz(self.dim.T).shape)
        
    def test_convert_to_dim(self):
        true = self.dim
        testing.assert_array_almost_equal(true, convert_to_dim(self.xyz))
        testing.assert_array_almost_equal(true.T, convert_to_dim(self.xyz.T))

        self.assertEqual(np.shape([1,2,3]), convert_to_dim([1,2,3]).shape)
        self.assertEqual(self.xyz.shape, convert_to_dim(self.xyz).shape)
        self.assertEqual(self.xyz.T.shape, convert_to_dim(self.xyz.T).shape)
        
    def test_rotate_around_axis(self):
        # test the shape of input and output array
        self.assertEqual(self.xyz.shape, rotate_around_axis(self.xyz, [1, 1, 1], theta=0).shape)
        self.assertEqual(self.xyz.T.shape, rotate_around_axis(self.xyz.T, [1, 1, 1], theta=0).shape)

        # test for theta = 0
        testing.assert_array_equal(self.xyz, rotate_around_axis(self.xyz, [1, 1, 1], theta=0))
        testing.assert_array_equal(self.xyz.T, rotate_around_axis(self.xyz.T, [1, 1, 1], theta=0))

        testing.assert_array_almost_equal(self.dim, rotate_around_axis(self.dim, [1, 1, 1], input='dim', theta=0))
        testing.assert_array_almost_equal(self.dim.T, rotate_around_axis(self.dim.T, [1, 1, 1], input='dim', theta=0))

    def test_rotate_arbitrary(self):
        # test the shape of input and output array
        self.assertEqual(self.xyz.shape, rotate_arbitrary(self.xyz, 1, 1, 1).shape)
        self.assertEqual(self.xyz.T.shape, rotate_arbitrary(self.xyz.T, 1, 1, 1).shape)
        self.assertEqual(self.dim.shape, rotate_arbitrary(self.dim, 1, 1, 1, input='dim').shape)
        self.assertEqual(self.dim.T.shape, rotate_arbitrary(self.dim.T, 1, 1, 1, input='dim').shape)

        # test for theta = 0
        testing.assert_array_equal(self.xyz, rotate_arbitrary(self.xyz, 0, 0, 0))
        testing.assert_array_equal(self.xyz.T, rotate_arbitrary(self.xyz.T, 0, 0, 0))

        testing.assert_array_almost_equal(self.dim, rotate_arbitrary(self.dim, 0, 0, 0, input='dim'))
        testing.assert_array_almost_equal(self.dim.T, rotate_arbitrary(self.dim.T, 0, 0, 0, input='dim'))

        # test for theta = 360
        testing.assert_array_almost_equal(self.xyz, rotate_arbitrary(self.xyz, 360, 0, 0))
        testing.assert_array_almost_equal(self.xyz.T, rotate_arbitrary(self.xyz.T, 360, 0, 0))

        testing.assert_array_almost_equal(self.dim, rotate_arbitrary(self.dim, 360, 0, 0, input='dim'))
        testing.assert_array_almost_equal(self.dim.T, rotate_arbitrary(self.dim.T, 360, 0, 0, input='dim'))

        testing.assert_array_almost_equal(self.xyz, rotate_arbitrary(self.xyz, 0, 360, 0))
        testing.assert_array_almost_equal(self.xyz.T, rotate_arbitrary(self.xyz.T, 0, 360, 0))

        testing.assert_array_almost_equal(self.dim, rotate_arbitrary(self.dim, 0, 360, 0, input='dim'))
        testing.assert_array_almost_equal(self.dim.T, rotate_arbitrary(self.dim.T, 0, 360, 0, input='dim'))

        testing.assert_array_almost_equal(self.xyz, rotate_arbitrary(self.xyz, 0, 0, 360))
        testing.assert_array_almost_equal(self.xyz.T, rotate_arbitrary(self.xyz.T, 0, 0, 360))

        testing.assert_array_almost_equal(self.dim, rotate_arbitrary(self.dim, 0, 0, 360, input='dim'))
        testing.assert_array_almost_equal(self.dim.T, rotate_arbitrary(self.dim.T, 0, 0, 360, input='dim'))

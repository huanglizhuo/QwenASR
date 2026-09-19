import unittest

import numpy as np

from convert_model import quantize_rows, quantize_cpu_decode_rows


class QuantizationTests(unittest.TestCase):
    def test_cpu_decode_truncates_instead_of_nearest_rounding(self):
        x = np.array([[127, -127, .5, 1.9, -.5, -1.9, 3.99, -3.99], [0]*8], dtype=np.float32)
        q, scales = quantize_cpu_decode_rows(x)
        np.testing.assert_array_equal(q, [[127, -127, 0, 1, 0, -1, 3, -3], [0]*8])
        np.testing.assert_array_equal(scales, [1, 1])

    def test_cpu_decode_rejects_unimplemented_scalar_tail(self):
        with self.assertRaises(ValueError):
            quantize_cpu_decode_rows(np.zeros((1, 7), dtype=np.float32))

    def test_exact_endpoints_and_declared_rounding(self):
        data = np.array([[127, -127, .5, 1.5, -.5, -1.5, 0]], dtype=np.float32)
        q, scales = quantize_rows(data)
        np.testing.assert_array_equal(q, [[127, -127, 0, 2, 0, -2, 0]])
        np.testing.assert_array_equal(scales, [1])

    def test_zero_row_and_independent_output_scales(self):
        data = np.array([[0, 0], [127, -63.5], [.127, -.0635]], dtype=np.float32)
        q, scales = quantize_rows(data)
        np.testing.assert_array_equal(q[0], [0, 0])
        self.assertEqual(scales[0], 1)
        np.testing.assert_allclose(scales[1:], [1, .001])
        self.assertTrue(np.isfinite(scales).all())

    def test_conv_rows_share_scale_across_channels_and_spatial_kernel(self):
        x = np.random.default_rng(37).normal(size=(7, 3, 3, 3)).astype(np.float32)
        q, scales = quantize_rows(x)
        self.assertEqual(q.shape, x.shape)
        restored = q.astype(np.float32) * scales[:, None, None, None]
        error = np.max(np.abs(x-restored), axis=(1, 2, 3))
        self.assertTrue(np.all(error <= scales*.50001))


if __name__ == "__main__":
    unittest.main()

from unittest import TestCase

import numpy as np

from test import rel_error
from two_layer_net.layer import affine_forward


class TestAffine(TestCase):

    def test_affine_forward(self):
        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)

        out, _ = affine_forward(x, w, b)
        correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                                [3.25553199, 3.5141327, 3.77273342]])

        # Compare your output with ours. The error should be around e-9 or less.
        print('Testing affine_forward function:')
        print('difference: ', rel_error(out, correct_out))

    def test(self):
        N, D, M = 2, 3, 4
        X = np.random.randn(N, D)
        W = np.random.randn(D, M)
        b = np.random.randn(M)
        out = X @ W + b
        self.assertEqual(out.shape, (N, M))

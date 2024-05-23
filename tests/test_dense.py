import unittest
import numpy as np
from src.DenseLayer import DenseLayer


class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        self.layer = DenseLayer(nb_Node=5, activation="relu", input_shape=3)

    def test_init(self):
        self.assertEqual(self.layer.nb_Node, 5)
        self.assertEqual(self.layer.activation.__name__, "relu")
        self.assertEqual(self.layer.input_shape, 3)
        self.assertEqual(self.layer.W.shape, (3, 5))
        self.assertEqual(self.layer.b.shape, (1, 5))

    def test_setInputShape(self):
        self.layer.setInputShape(4)
        self.assertEqual(self.layer.input_shape, 4)
        self.assertEqual(self.layer.W.shape, (4, 5))

    def test_forward(self):
        a_in = np.array([[1, 2, 3]])
        output = self.layer.forward(a_in)
        self.assertEqual(output.shape, (1, 5))

    def test_setErrorLastLayer(self):
        self.layer.output = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        y = np.array([2])
        self.layer.setErrorLastLayer(y)
        self.assertEqual(self.layer.errors.shape, (1, 5))
        self.assertEqual(self.layer.delta.shape, (1, 5))

    def test_setError(self):
        next_layer = DenseLayer(nb_Node=3, input_shape=5)
        next_layer.delta = np.array([[0.1, 0.2, 0.3]])
        self.layer.output = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        self.layer.setError(next_layer)
        self.assertEqual(self.layer.errors.shape, (1, 5))
        self.assertEqual(self.layer.delta.shape, (1, 5))

    def test_setActivation(self):
        self.layer.setActivation("sigmoid")
        self.assertEqual(self.layer.activation.__name__, "sigmoid")
        self.assertEqual(self.layer.activation_deriv.__name__, "sigmoid_deriv")

    def test_setWeightInit(self):
        self.layer.setWeightInit("xavierUniform")
        self.assertEqual(self.layer.weights_initializer.__name__, "xavierUniformWI")

    def test_setLayerName(self):
        self.layer.setLayerName("hidden_layer")
        self.assertEqual(self.layer.layerName, "hidden_layer")

if __name__ == '__main__':
    unittest.main()
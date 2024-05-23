import unittest
import numpy as np
from src.MySequencial import MySequencial
from src.DenseLayer import DenseLayer

class TestMySequencial(unittest.TestCase):
    def setUp(self):
        self.layers = [
            DenseLayer(nb_Node=5, activation="relu", input_shape=3),
            DenseLayer(nb_Node=2, activation="sigmoid")
        ]
        self.model = MySequencial(self.layers)

    def test_init(self):
        self.assertEqual(len(self.model.layers), 2)
        self.assertEqual(self.model.layers[0].layerName, "Dense 1")
        self.assertEqual(self.model.layers[1].layerName, "Dense 2")

    def test_normaliseData(self):
        data = np.array([[1, 2, 3, 0], [4, 5, 6, 1]])
        normalized_data = self.model.normaliseData(data, training=True)
        self.assertEqual(normalized_data.shape, (2, 4))

    def test_getSequentialArchitecture(self):
        architecture = self.model.getSequentialArchitecture()
        expected_architecture = "5 relu heUniformWI | 2 sigmoid heUniformWI"
        self.assertEqual(architecture, expected_architecture)

    def test_predict(self):
        X = np.array([[1, 2, 3]])
        output = self.model.predict(X)
        self.assertEqual(output.shape, (1, 2))

    def test_accuracy(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([0, 1])
        accuracy = self.model.accuracy(X, Y)
        self.assertIsInstance(accuracy, float)

    def test_precisionAndRecall(self):
        X = np.array([[1, 0, 0], [4, 5, 6]])
        y_val = np.array([1, 1])
        precision, recall = self.model.precisionAndRecall(X, y_val)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)

if __name__ == '__main__':
    unittest.main()
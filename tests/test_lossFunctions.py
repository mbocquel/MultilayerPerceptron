import unittest
import numpy as np
from src.LossFunctions import isLossFunction, binaryCrossentropyLoss, sparseCategoricalCrossEntropyLoss, setLossFunction

class TestLossFunctions(unittest.TestCase):
    
    def test_isLossFunction(self):
        self.assertTrue(isLossFunction("binaryCrossentropyLoss"))
        self.assertTrue(isLossFunction("sparseCategoricalCrossEntropyLoss"))
        self.assertFalse(isLossFunction("unknownLossFunction"))

    def test_binaryCrossentropyLoss(self):
        class MockModel:
            def predict(self, x):
                return np.array([[0.2, 0.8], [0.3, 0.7]])

        model = MockModel()
        x = np.array([[1, 2], [3, 4]])
        y = np.array([1, 0])

        loss = binaryCrossentropyLoss(model, x, y)
        self.assertAlmostEqual(loss, 0.7136, places=4)

    def test_sparseCategoricalCrossEntropyLoss(self):
        class MockModel:
            def predict(self, x):
                return np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])

        model = MockModel()
        x = np.array([[1, 2], [3, 4]])
        y = np.array([2, 1])

        loss = sparseCategoricalCrossEntropyLoss(model, x, y)
        self.assertAlmostEqual(loss, 0.5249, places=4)

    def test_setLossFunction(self):
        loss_function = setLossFunction("binaryCrossentropyLoss")
        self.assertEqual(loss_function, binaryCrossentropyLoss)

        loss_function = setLossFunction("sparseCategoricalCrossEntropyLoss")
        self.assertEqual(loss_function, sparseCategoricalCrossEntropyLoss)

        with self.assertRaises(ValueError):
            setLossFunction("unknownLossFunction")

if __name__ == '__main__':
    unittest.main()
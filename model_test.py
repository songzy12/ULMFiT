import unittest

import numpy as np
import paddle

from model import get_language_model, get_text_classifier


class TestLanguageModel(unittest.TestCase):
    def test_language_model(self):
        VOCAB_SIZE = 10000
        HIDDEN_SIZE = 650
        BATCH_SIZE = 20
        NUM_LAYERS = 2
        DROPOUT = 0.5

        network = get_language_model(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT)
        print(network)

        NUM_STEPS = 35
        x_data = np.random.randint(0, VOCAB_SIZE, (
            BATCH_SIZE, NUM_STEPS)).astype('int64')  # [BATCH_SIZE, NUM_STEPS]
        x = paddle.to_tensor(x_data)
        y = network.forward(x)
        assert (y.shape == [BATCH_SIZE, NUM_STEPS, VOCAB_SIZE])


class TestTextClassifier(unittest.TestCase):
    def test_text_classifier(self):
        VOCAB_SIZE = 10000
        HIDDEN_SIZE = 650
        BATCH_SIZE = 20
        NUM_LAYERS = 2
        DROPOUT = 0.5
        N_CLASS = 2

        network = get_text_classifier(
            vocab_size=VOCAB_SIZE,
            n_class=N_CLASS,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT)
        print(network)

        NUM_STEPS = 35
        x_data = np.random.randint(0, VOCAB_SIZE, (
            BATCH_SIZE, NUM_STEPS)).astype('int64')  # [BATCH_SIZE, NUM_STEPS]
        x = paddle.to_tensor(x_data)
        y = network.forward(x)
        assert (y.shape == [BATCH_SIZE, N_CLASS])


if __name__ == '__main__':
    unittest.main()

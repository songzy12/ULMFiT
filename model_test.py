import unittest

import numpy as np
import paddle

from model import RnnLm


class TestRnnLm(unittest.TestCase):
    def test_rnn_lm(self):
        VOCAB_SIZE = 10000
        HIDDEN_SIZE = 650
        BATCH_SIZE = 20
        NUM_LAYERS = 2
        INIT_SCALE = 0.05
        DROPOUT = 0.5

        network = RnnLm(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            num_layers=NUM_LAYERS,
            init_scale=INIT_SCALE,
            dropout=DROPOUT)
        print(network)

        NUM_STEPS = 35
        x_data = np.random.randint(0, VOCAB_SIZE, (
            BATCH_SIZE, NUM_STEPS)).astype('int64')  # [BATCH_SIZE, NUM_STEPS]
        x = paddle.to_tensor(x_data)
        y = network.forward(x)
        assert (y.shape == [BATCH_SIZE, NUM_STEPS, VOCAB_SIZE])


if __name__ == '__main__':
    unittest.main()
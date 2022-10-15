from cgitb import text
import unittest

import numpy as np
import paddle

from model import get_language_model, get_text_classifier, load_encoder, save_encoder

VOCAB_SIZE = 10000
HIDDEN_SIZE = 650
BATCH_SIZE = 20
NUM_LAYERS = 2
DROPOUT = 0.5
N_CLASS = 2


class TestLanguageModel(unittest.TestCase):

    def test_language_model(self):
        network = get_language_model(vocab_size=VOCAB_SIZE,
                                     hidden_size=HIDDEN_SIZE,
                                     batch_size=BATCH_SIZE,
                                     num_layers=NUM_LAYERS,
                                     dropout=DROPOUT)
        # print(network)

        NUM_STEPS = 35
        x_data = np.random.randint(0,
                                   VOCAB_SIZE, (BATCH_SIZE, NUM_STEPS)).astype(
                                       'int64')  # [BATCH_SIZE, NUM_STEPS]
        x = paddle.to_tensor(x_data)
        y = network.forward(x)
        assert (y.shape == [BATCH_SIZE, NUM_STEPS, VOCAB_SIZE])


class TestTextClassifier(unittest.TestCase):

    def test_text_classifier(self):
        network = get_text_classifier(vocab_size=VOCAB_SIZE,
                                      n_class=N_CLASS,
                                      hidden_size=HIDDEN_SIZE,
                                      batch_size=BATCH_SIZE,
                                      num_layers=NUM_LAYERS,
                                      dropout=DROPOUT)
        # print(network)

        NUM_STEPS = 35
        x_data = np.random.randint(0,
                                   VOCAB_SIZE, (BATCH_SIZE, NUM_STEPS)).astype(
                                       'int64')  # [BATCH_SIZE, NUM_STEPS]
        x = paddle.to_tensor(x_data)
        y = network.forward(x)
        assert (y.shape == [BATCH_SIZE, N_CLASS])


class TestSaveAndLoadEncoder(unittest.TestCase):

    def test_save_and_load_encoder(self):
        encoder_filename = "/tmp/encoder.pdparams"

        language_model = get_language_model(vocab_size=VOCAB_SIZE,
                                            hidden_size=HIDDEN_SIZE,
                                            batch_size=BATCH_SIZE,
                                            num_layers=NUM_LAYERS,
                                            dropout=DROPOUT)
        save_encoder(language_model, encoder_filename)

        text_classifier = get_text_classifier(vocab_size=VOCAB_SIZE,
                                              n_class=N_CLASS,
                                              hidden_size=HIDDEN_SIZE,
                                              batch_size=BATCH_SIZE,
                                              num_layers=NUM_LAYERS,
                                              dropout=DROPOUT)
        load_encoder(text_classifier, encoder_filename)

        encoder1_state_dict = language_model['encoder'].state_dict()
        encoder2_state_dict = text_classifier['encoder'].state_dict()

        for name, weight1 in encoder1_state_dict.items():
            assert name in encoder2_state_dict.keys()
            weight2 = encoder2_state_dict[name]
            assert weight1.shape == weight2.shape
            assert paddle.equal_all(weight1, weight2)


if __name__ == '__main__':
    unittest.main()

import unittest

from reader import create_data_loader


class TestDataLoader(unittest.TestCase):

    def test_create_data_loader(self):
        BATCH_SIZE = 20
        NUM_STEPS = 35
        train_loader, valid_loader, test_loader, vocab_size = create_data_loader(
            batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

        assert (vocab_size == 10000)

        assert (len(train_loader) == 1327)
        assert (len(valid_loader) == 105)
        assert (len(test_loader) == 117)

        data, label = next(train_loader())
        assert (data.shape == [BATCH_SIZE, NUM_STEPS])
        assert (label.shape == [BATCH_SIZE, NUM_STEPS])

        # label[0] == data[1], label[1] == data[2], etc.
        assert (data[0][1:].equal_all(label[0][:-1]))


if __name__ == '__main__':
    unittest.main()

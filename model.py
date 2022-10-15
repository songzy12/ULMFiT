import paddle
import paddle.nn as nn
import paddle.nn.initializer as I


class AWD_LSTM(nn.Layer):
    # fastai/fastai/text/models/awdlstm.py
    def __init__(self, vocab_size, hidden_size, batch_size, num_layers,
                 dropout):
        super(AWD_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.reset()

        self.embedder = nn.Embedding(vocab_size, hidden_size)

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        x = inputs  # [batch_size, num_steps]
        x_emb = self.embedder(x)  # [batch_size, num_steps, hidden_size]
        x_emb = self.dropout(x_emb)  # [batch_size, num_steps, hidden_size]

        y, (self.hidden, self.cell) = self.lstm(
            x_emb, (self.hidden, self.cell)
        )  # y.shape == [batch_size, num_steps, hidden_size]; hidden.shape == cell.shape == [num_layers, batch_size, hidden_size]
        (self.hidden, self.cell) = tuple(
            [item.detach() for item in (self.hidden, self.cell)])
        y = self.dropout(y)  # y.shape == [batch_size, num_steps, hidden_size]
        return y

    def reset(self):
        # TODO(songzy): reset should not depend on batch_size.
        self.hidden = paddle.zeros(
            shape=[self.num_layers, self.batch_size, self.hidden_size],
            dtype='float32')
        self.cell = paddle.zeros(
            shape=[self.num_layers, self.batch_size, self.hidden_size],
            dtype='float32')


class PoolingLinearClassifier(nn.Layer):

    def __init__(self, in_features, out_features):
        super(PoolingLinearClassifier, self).__init__()

        # TODO(songzy): add more layers.
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = paddle.mean(x, axis=1)
        y = self.linear(x)
        return y


class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."

    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'):
                getattr(c, 'reset')()


def get_language_model(vocab_size, hidden_size, batch_size, num_layers,
                       dropout):
    # fastai/fastai/text/models/core.py
    encoder = AWD_LSTM(vocab_size, hidden_size, batch_size, num_layers,
                       dropout)
    decoder = nn.Linear(hidden_size, vocab_size)
    return SequentialRNN(("encoder", encoder), ("decoder", decoder))


def get_text_classifier(vocab_size, n_class, hidden_size, batch_size,
                        num_layers, dropout):
    # fastai/text/models/core.py
    encoder = AWD_LSTM(vocab_size, hidden_size, batch_size, num_layers,
                       dropout)
    decoder = PoolingLinearClassifier(hidden_size, n_class)
    return SequentialRNN(("encoder", encoder), ("decoder", decoder))


def save_encoder(model, filename):
    assert 'encoder' in dir(model)
    encoder = model['encoder']

    paddle.save(encoder.state_dict(), filename)


def load_encoder(model, filename):
    assert 'encoder' in dir(model)
    encoder = model['encoder']

    model_dict = encoder.state_dict()
    model_state_dict = paddle.load(filename)
    # The following section is for debugging purpose only.
    for name, weight in model_dict.items():
        if name in model_state_dict.keys():
            if weight.shape != list(model_state_dict[name].shape):
                print(
                    '{} not used, shape {} unmatched with {} in model.'.format(
                        name, list(model_state_dict[name].shape),
                        weight.shape))
                model_state_dict.pop(name, None)
        else:
            print('Lack weight: {}'.format(name))

    encoder.set_dict(model_state_dict)

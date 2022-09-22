import paddle
import paddle.nn as nn
import paddle.nn.initializer as I


class AWD_LSTM(nn.Layer):
    # fastai/fastai/text/models/awdlstm.py
    def __init__(self, vocab_size, hidden_size, batch_size, num_layers,
                 init_scale, dropout):
        super(AWD_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.batch_size = batch_size
        self.reset()

        self.embedder = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_ih_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)),
            weight_hh_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

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


class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."

    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'):
                getattr(c, 'reset')()


def get_language_model(vocab_size, hidden_size, batch_size, num_layers,
                       init_scale, dropout):
    # fastai/fastai/text/models/core.py
    encoder = AWD_LSTM(vocab_size, hidden_size, batch_size, num_layers,
                       init_scale, dropout)
    decoder = nn.Linear(
        hidden_size,
        vocab_size,
        weight_attr=paddle.ParamAttr(
            initializer=I.Uniform(low=-init_scale, high=init_scale)),
        bias_attr=paddle.ParamAttr(
            initializer=I.Uniform(low=-init_scale, high=init_scale)))
    return SequentialRNN(("encoder", encoder), ("decoder", decoder))

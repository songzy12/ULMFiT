import paddle
import paddle.nn as nn
import paddle.nn.initializer as I


class RnnLm(nn.Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=0.0):
        super(RnnLm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.batch_size = batch_size
        self.reset_states()

        self.embedder = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_ih_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            weight_hh_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        self.fc = nn.Linear(
            hidden_size,
            vocab_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

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
        y = self.fc(y)  # y.shape == [batch_size, num_steps, vocab_size]
        return y

    def reset_states(self):
        self.hidden = paddle.zeros(
            shape=[self.num_layers, self.batch_size, self.hidden_size],
            dtype='float32')
        self.cell = paddle.zeros(
            shape=[self.num_layers, self.batch_size, self.hidden_size],
            dtype='float32')

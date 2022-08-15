LSTM language model in Paddle: 

- <http://www.fit.vutbr.cz/~imikolov/rnnlm/>
- <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/rnnlm>
- <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html>

LSTM text classification model in Paddle:

- <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/rnn>

So, what is the different between 
- `LSTMEncoder` in `paddlenlp/seq2vec/encoder.py`
- `LSTM` in `paddle/nn/layer/rnn.py`

### Language Model

`PaddleNLP/examples/language_model/rnnlm/model.py`

```
class RnnLm(nn.Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=0.0):
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_ih_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)),
            weight_hh_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

    def forward(self, inputs):
        y, (self.hidden, self.cell) = self.lstm(x_emb, (self.hidden, self.cell))
```

### Text Classification

`PaddleNLP/examples/text_classification/rnn/model.py`

```
class LSTMModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):

        self.lstm_encoder = nlp.seq2vec.LSTMEncoder(emb_dim,
                                                    lstm_hidden_size,
                                                    num_layers=lstm_layers,
                                                    direction=direction,
                                                    dropout=dropout_rate,
                                                    pooling_type=pooling_type)

    def forward(self, text, seq_len):
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
```

`paddlenlp/seq2vec/encoder.py`

```
class LSTMEncoder(nn.Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction=direction,
            dropout=dropout,
            **kwargs)

    def forward(self, inputs, sequence_length):
        encoded_text, (last_hidden, last_cell) = self.lstm_layer(
            inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            if self._direction != 'bidirect':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
```

import paddle
import paddle.nn as nn


class CrossEntropyLossForLm(nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForLm, self).__init__()

    def forward(self, y, label):
        # y.shape == [batch_size, num_steps, vocab_size]
        # label.shape == [batch_size, num_steps]
        label = paddle.unsqueeze(
            label, axis=2)  # label.shape == [batch_size, num_steps, 1]
        loss = paddle.nn.functional.cross_entropy(
            input=y, label=label,
            reduction='none')  # loss.shape == [batch_size, num_steps, 1]
        loss = paddle.squeeze(
            loss, axis=[2])  # loss.shape == [batch_size, num_steps]
        loss = paddle.mean(loss, axis=[0])  # loss.shape == [num_steps]
        loss = paddle.sum(loss)  # loss.shape == [1]
        return loss

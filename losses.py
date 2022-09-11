import paddle
import paddle.nn as nn


class CrossEntropyLossForLm(nn.Layer):

    def __init__(self):
        super(CrossEntropyLossForLm, self).__init__()

    def forward(self, y, label):
        label = paddle.unsqueeze(label, axis=2)
        loss = paddle.nn.functional.cross_entropy(input=y,
                                                  label=label,
                                                  reduction='none')
        loss = paddle.squeeze(loss, axis=[2])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)
        return loss

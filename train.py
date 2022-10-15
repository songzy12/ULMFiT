import paddle

from args import parse_args
from data_loader import create_data_loader_for_lm, create_data_loader_for_text_classifier
from model import get_language_model, get_text_classifier
from callbacks import ResetModel
from losses import CrossEntropyLossForLm

from paddlenlp.metrics import Perplexity

paddle.seed(102)


def pretrain_lm(args):
    train_loader, valid_loader, test_loader, vocab_size = create_data_loader_for_lm(
        batch_size=args.batch_size, num_steps=args.num_steps)

    network = get_language_model(vocab_size=vocab_size,
                                 hidden_size=args.hidden_size,
                                 batch_size=args.batch_size,
                                 num_layers=args.num_layers,
                                 dropout=args.dropout)
    model = paddle.Model(network)

    learning_rate = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.base_lr,
        lr_lambda=lambda x: args.lr_decay**max(x + 1 - args.epoch_start_decay,
                                               0.0),
        verbose=True)
    gloabl_norm_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate,
                                     parameters=model.parameters(),
                                     grad_clip=gloabl_norm_clip)

    model.prepare(optimizer=optimizer,
                  loss=CrossEntropyLossForLm(),
                  metrics=Perplexity())

    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    scheduler = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)
    benchmark_logger = paddle.callbacks.ProgBarLogger(log_freq=100, verbose=3)
    callbacks = [ResetModel(), scheduler, benchmark_logger]

    model.fit(train_data=train_loader,
              eval_data=valid_loader,
              epochs=args.max_epoch,
              shuffle=False,
              callbacks=callbacks)

    model.save(path='checkpoint/test')  # save for training

    print('Start to evaluate on test dataset...')
    model.evaluate(test_loader, log_freq=len(test_loader))


def train_text_classifier(args):
    train_loader, valid_loader, test_loader, vocab_size = create_data_loader_for_text_classifier(
        batch_size=args.batch_size, num_steps=args.num_steps)

    network = get_text_classifier(vocab_size=vocab_size,
                                  n_class=2,
                                  hidden_size=args.hidden_size,
                                  batch_size=args.batch_size,
                                  num_layers=args.num_layers,
                                  dropout=args.dropout)
    model = paddle.Model(network)

    learning_rate = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.base_lr,
        lr_lambda=lambda x: args.lr_decay**max(x + 1 - args.epoch_start_decay,
                                               0.0),
        verbose=True)
    gloabl_norm_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate,
                                     parameters=model.parameters(),
                                     grad_clip=gloabl_norm_clip)

    model.prepare(optimizer=optimizer,
                  loss=paddle.nn.CrossEntropyLoss(),
                  metrics=paddle.metric.Accuracy())

    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    scheduler = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)
    benchmark_logger = paddle.callbacks.ProgBarLogger(log_freq=100, verbose=3)
    callbacks = [ResetModel(), scheduler, benchmark_logger]

    model.fit(train_data=train_loader,
              eval_data=valid_loader,
              epochs=args.max_epoch,
              shuffle=False,
              callbacks=callbacks)

    model.save(path='checkpoint/test')  # save for training

    # NOTE: the test_loader does not have 'labels' info.
    print('Start to evaluate on valid dataset...')
    model.evaluate(valid_loader, log_freq=len(valid_loader))


if __name__ == '__main__':
    args = parse_args()
    paddle.set_device(args.device)
    pretrain_lm(args)
    # TODO(songzy): save encoder after pretrain lm, and load encoder before train text classifier.
    train_text_classifier(args)

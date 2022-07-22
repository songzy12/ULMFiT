fastai/nbs/38_tutorial.text.ipynb

```
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], path=path, wd=0.1).to_fp16()
learn.fit_one_cycle(1, 1e-2)
learn.save_encoder('finetuned')

learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn = learn.load_encoder('finetuned')
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
```

So the connection of the language model and the text classifier is a **common encoder**.

fastai/nbs/33_text.models.core.ipynb

```
def get_language_model(
    arch, # Function or class that can generate a language model architecture
    vocab_sz:int, # Size of the vocabulary
    config:dict=None, # Model configuration dictionary
    drop_mult:float=1. # Multiplicative factor to scale all dropout probabilities in `config`
) -> SequentialRNN: # Language model with `arch` encoder and linear decoder
    "Create a language model from `arch` and its `config`."

    encoder = arch(vocab_sz, **config)
    decoder = LinearDecoder(vocab_sz, config[meta['hid_name']], output_p, tie_encoder=enc, bias=out_bias)
    model = SequentialRNN(encoder, decoder)
```

```
def get_text_classifier(
    arch:callable, # Function or class that can generate a language model architecture
    vocab_sz:int, # Size of the vocabulary
    n_class:int, # Number of classes
    seq_len:int=72, # Backpropagation through time
    config:dict=None, # Encoder configuration dictionary
    drop_mult:float=1., # Multiplicative factor to scale all dropout probabilities in `config`
    lin_ftrs:list=None, # List of hidden sizes for classifier head as `int`s
    ps:list=None, # List of dropout probabilities for classifier head as `float`s
    pad_idx:int=1, # Padding token id
    max_len:int=72*20, # Maximal output length for `SentenceEncoder`
    y_range:tuple=None # Tuple of (low, high) output value bounds
):
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`"

    encoder = SentenceEncoder(seq_len, arch(vocab_sz, **config), pad_idx=pad_idx, max_len=max_len)
    model = SequentialRNN(encoder, PoolingLinearClassifier(layers, ps, bptt=seq_len, y_range=y_range))
```

So the difference between the language model and the text classifier is 

1. LinearDecoder
2. PoolingLinearClassifier

fastai/nbs/38_tutorial.text.ipynb

- language_model_learner
- text_classifier_learner

fastai/fastai/text/models/core.py

- get_language_model
- get_text_classifier

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

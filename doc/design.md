ULMFiT contains 3 steps:

1. general domain LM pretraining
2. target task LM fine-tuning
3. target task classifier fine-tuning

AWD-LSTM is the backbone used by ULMFiT, where AWD-LSTM stands for *ASGD Weight-Dropped LSTM*.

Instead of AWD-LSTM, let us start with a vanilla LSTM.

Datasets:

1. LM: Wikitext-103
2. Topic classification: AG news

To start with, we can also skip step 2, i.e., the target task LM fine-tuning:

1. general domain LM pretraining
3. target task classifier fine-tuning

[1] Universal Language Model Fine-tuning for Text Classification.

[2] Regularizing and Optimizing LSTM Language Models.

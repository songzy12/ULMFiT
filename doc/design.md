## ULMFiT

### Overview

ULMFiT contains 3 steps:

1. general domain LM pretraining
2. target task LM fine-tuning
3. target task classifier fine-tuning

AWD-LSTM is the backbone used by ULMFiT, where AWD-LSTM stands for *ASGD Weight-Dropped LSTM*.

### Datasets

1. LM: Wikitext-103
2. Topic classification: AG news

### Detail

#### M1. General Domain LM Pretraining

Instead of AWD-LSTM, let us start with a vanilla LSTM.

#### M2. Target Task LM Fine-tuning

#### M3. Target Task Classifier Fine-tuning

### Reference

1. Universal Language Model Fine-tuning for Text Classification.
2. Regularizing and Optimizing LSTM Language Models.

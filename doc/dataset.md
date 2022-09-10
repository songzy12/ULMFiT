Given we do not have wikitext 103 and AG news in paddle dataset, we start with PTB for LM and SST-2 for classification.

<https://github.com/songzy12/ULMFiT-paddle/issues/7> filed for this issue.

### PTB

<https://paperswithcode.com/dataset/penn-treebank>

> The English Penn Treebank (PTB) corpus, and in particular the section of the corpus corresponding to the articles of Wall Street Journal (WSJ), is one of the most known and used corpus for the evaluation of models for sequence labelling. The task consists of annotating each word with its Part-of-Speech tag. In the most common split of this corpus, sections from 0 to 18 are used for training (38 219 sentences, 912 344 tokens), sections from 19 to 21 are used for validation (5 527 sentences, 131 768 tokens), and sections from 22 to 24 are used for testing (5 462 sentences, 129 654 tokens). The corpus is also commonly used for character-level and word-level Language Modelling.

Similar datasets:
- <https://paperswithcode.com/dataset/wikitext-2>
- <https://paperswithcode.com/dataset/wikitext-103>

SOTA: 
- <https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word>

### SST-2

<https://paperswithcode.com/dataset/sst>

> The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.

> Each phrase is labelled as either negative, somewhat negative, neutral, somewhat positive or positive. The corpus with all 5 labels is referred to as SST-5 or SST fine-grained. Binary classification experiments on full sentences (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) refer to the dataset as SST-2 or SST binary.

SOTA:
- <https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary>

### Reference

1. https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html
---
layout: post
title: All we need is tokenizerless
author: nicolas
toc: true
description: Going beyond transformers
tags:
  - ml
  - tokenizerless
  - mamba
---

# What tokenization achieves

Tokenization enables models to have some form of temporal compression (read outputting several characters as 1 timestep) while also increasing the entropy of those item. By definition we use compression algorithm to create the tokens, which means the distribution is trying to become closer the the uniform distribution.

That means that models outputting random tokens will still make a coherent native looking output. It also means that the random distribution is a good starting point for the model itself (which wouldn't be the case with any real language)

# Tokenization limits


Tokenization is limiting because it doesn't take into account previous tokens. Therefore, while the distribution of tokens is relatively uniform, the distribution of pairs of tokens is most definitely not, so some pairs are easier to train than others. This is what makes speculation working so well. One would think than having a more complex tokenizer would help here, but having more tokens in the vocabulary also is a cost, and because the tokenization methods are so crude, they yield diminishing returns.

Also tokenization doesn't really work well with unicode. Unicode is the method to create all non ascii characters, where most non latin languages rely on. For a language like Chinese, you would need a vocabulary of size ~100k to just to get all the initial vocabulary (10k being a bare minimum). Because that's humongous and some of these bare characters are underrepresented, usually tokenizers remove some of these rare base chars, creating UNK tokens once again. The other technique is called byte fallback, meaning those unknown characters are shown to the model as their raw bytes representations. This works in training, but that also means that the model can output arbitrary sequences of raw bytes, meaning the model can also talk non valid utf-8 outputs which you have to deal with.

# Temporal coherence

One thing to note, is that if you look at the output distribution of a transformers, even a simple one, there are definite breakpoints in *entropy*, and they really look like the original tokenizers, like "the ", or "or ".


# Multi level hierarchy

Achieving tokenizerless temporal coherence by pure ML ways, would enable models to stack various levels of temporal coherence on top of each other. There could be one level for the "tokens", but also some for the "sentences", or "paragraphs" or even "entire books".

This is actually how most written content is written. Not exactly word for word in a continuous stream, but more a back and forth between multiple timescales. Write the abstract, expand some paragraphs, revisit the structure, rewrite sentences, fix typos, go back again to the top level structure, change a bit the conclusion, create a title, go back down to the first paragraph and so on. There is a constant jumping between timescale representations that produces modifications to another timescale representation.


# Results






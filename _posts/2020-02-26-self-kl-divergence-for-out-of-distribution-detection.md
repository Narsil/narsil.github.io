---
layout: post
title: 'Self KL-divergence for detecting out of distribution data and unsupervised text classification'
author: nicolas
---

> TL;DR. By training two models at the same time (same architecture, same loss, but different initialization)
> I was able to obtain a consistent out-of-distribution detector by measuring the kl-divergence between model outputs.
> This out-of-distribution measure used on text could lead to unsupervised text classification.

1/ What's the problem ?

ML models usually are not really capable of predicting how well the data you
feed them is close to what was in the dataset. It really matters in production
models as they might make really stupid mistakes just because they are off
the training set.

- [ ] Find a good example of failure
- [x] Find classical solutions to this problem.
- [ ] Good thing that makes this measure better than other methods is that it's measured in bits/nats so you know.
- [ ] Less tl;dr as to why this should work (lottery ticket hypothesis, model structure forces some relevant manifold of the output

- https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html
- https://arxiv.org/pdf/1910.04241.pdf (Manifold approximation via Embedding (VAE or GAN) and out-of-distribution sampling via Manifold perturbation.)
- https://paperswithcode.com/task/out-of-distribution-detection
- https://openreview.net/pdf?id=Hkxzx0NtDB (Energy based model, hard to train but effective, learn `p(x, y)` at the same time as `p(y|x)`)
- https://arxiv.org/pdf/1802.04865v1.pdf (Adding extra loss making optimization problem joint between confidence and accuracy)

2/ How do we solve it ?

Tl;dr : Make two similar models, with two different random initialization, then train them at the same time.
Check their converged average kl-distance on the train set, that will give you a baseline of what similar is.

Check what kind of values do you get on test/validation set. You should get something similar or higher.

Then you have by measuring this self kl-divergence on new sample a measure of newness. Then it's a matter of choosing your
own threshold about what's acceptable or not.

linked to the training data leading to good properties in terms of out of distribution values).

3/ Experiments

- Test two identical networks. With same training we should have kl-divergence = 0 everywhere. So no possibility of detecting out of distribution.
  Test on widely different architecture and check that we don't get correct results
- On same architecture, different initialization show that it can be used for out of distribution detection for english, french and the train set.
- Test with various initialization patterns, with various architectures. Show that it's linked to
  descent method, and probably structure of network (does not seem fully generalizable)
- Test with random inputs to check that it works.
- Test with adversarial sampling to see if we can generate samples from it.

4/ Unsupervised text classification

- Show that small network trained on a single english book enables to detect different languages
  or different patterns of writing (old english, irish, french, or event dictionnaries)
- The detection is super fined grained capable of detecting english within a French book.

5/ Limits

6/ Future work

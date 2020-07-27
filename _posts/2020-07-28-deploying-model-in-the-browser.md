---
layout: post
title: 'Deploying a model to the browser'
author: nicolas
toc: true
description: How to deploy a ML model without worrying about a cloud pipeline.
tags: [ml, onnjx, browser]
---

> TL;DR In this article I explain how you can deploy a model directly to the browser from pytorch by using Onnjx. This work was done a year ago.

[Check out the full demo](https://narsil.github.io/assets/face/).
**TODO** import GIF + link to the project.

## Deploying a cool deep learning demo at zero cost.

Ok, so when we are showcasing deep learning, usually that implies running models somewhere on the cloud. Sometimes, running these models is by itself quite costly. GPT-3 cost something like 10 million to train, but imagine how much it will cost to _run_ if it was accessible to the general public !

One technique applicable so small machine learning models, it to actually make to client run the model not you. This means that your front can be a simple static website. Hell you could even host it on Github for free !

## Background

About a year ago, I was working at [Nabla](https://nabla.com). We were looking
at how performant was 3d pose estimation of the face. It means models detecting
faces _with_ depth which was not as ubiquitous as regular 2d face detection.

The idea was to see how hard it was to fit glasses on the fly to customers.
The whole thing lasted for 2 weeks, so mind the lack of polish.

## Let's get started with 3DDFA

So [3DDFA](https://github.com/cleardusk/3DDFA) is an improved Pytorch implementation of [this paper](https://arxiv.org/abs/1804.01005).
We settled on that implementation because it was the best available at the time.

### How does it work ?

Simply enoughgtg

## Let's port the model to the browser.

## Putting that model in an actual demo product.

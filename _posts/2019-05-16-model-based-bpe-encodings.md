---
layout: post
title: 'Model based encodings'
author: nicolas
toc: true
description: How to use BPE without this hardcoded algorithm
categories: [ml, nlp]
---

[Byte-pair encodings](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE) are now very commonly used in NLP. In [GPT-2](https://openai.com/blog/better-language-models/), Byte-pair encodings are used to preformat the raw texts before feeding the model. But this is a relatively costly step for your preprocessing and has some limitations. For instance, you have to split your data on spaces if you want your byte pair algorithm to compute in reasonable time.

> # TL;DR In this article we present an idea to generate Byte pair encodings, not based on frequency in the dataset, but on the quality of the prediction of our model. This enables us to predict multi word tokens like "New York" and address languages that don't use spaces to split words.

## What are Byte Pair Encodings ?

Byte-pair encodings are a way to compress information from pairs of bytes that will form tokens. Let's take an example :

"I love carrots and I love apples."

This sentence read by a computer is only a sequence of bytes (bytes are simply a number between 0 and 255). That means to a computer our sentence looks like

"I love carrots and I love apples." -> [73, 32, 108, 111, 118, 101, 32, 99, 97, 114, 114, 111, 116, 115, 32, 97, 110, 100, 32, 73, 32, 108, 111, 118, 101, 32, 97, 112, 112, 108, 101, 115, 46]

From that example, you may remark that some bytes are occurring multiple times together like [108, 111] that occurs twice (it's "lo" from "love"). So let's build a new token for this frequent pair. Numbers from 0 to 255 are already taken so we'll take the next available number which is 256, and we are going to store that information in a table

[108, 111] -> 256

Now if we use that new token to encode our original bytes, whenever we encounter [108, 111], we'll replace that by 256, so the original byte string becomes :

[73, 32, 108, **256**, 101, 32, 99, 97, 114, 114, 111, 116, 115, 32, 97, 110, 100, 32, 73, 32, **256**, 118, 101, 32, 97, 112, 112, 108, 101, 115, 46]

We went from 33 numbers to 31 numbers. We can rinse and repeat to compress the number of numbers even further. Originally, BPE was proposed as a compression algorithm. It's not the best compression tool, so we won't look at that side of the algorithm. Now you get what we are looking at when we train a model on BPEs, just a list of numbers.

Typically a BPE vocabulary contains ~10k tokens (GPT-2 has 50k), that means it can capture very frequent words like "the" entirely, and parts of words that contain many variations like "ment" (**ment**ally, environ**ment** …). What's great about it it that you can now have words share semantic parts of them for their representation in your model so (environ-ment, environ-ment-al, environ-ment-ally will all share "environ" which will contain most of the semantic meaning, the rest will contain grammar information hopefully).

The real advantage of BPE over classical Word Embeddings is that it does not fall into the out-of-vocabulary error (when a word was not seen). At worse you can always fall back to single bytes.

## **What’s the problem with BPE ?**

BPE algorithm is pretty bad in terms of complexity to calculate (roughly O(n²), you can look at a very good implementation [https://github.com/glample/fastBPE](https://github.com/glample/fastBPE)). BPE is also pretty bad when you want to encode some new text. A greedy algorithm will be O(n) but not the best encoding possible, the best encoding possible is actually O(n²) in the general case.

To be honest, most implementations split on spaces as mentioned earlier which speeds up the algorithm quite a bit. Once we have encoded a full word like “the” there is no way to add tokens to it, so it’s not necessary to look at it anymore for potential byte pairs, so we can assume the encoding&table creation go from O(n²) to something much closer to O(n). In addition, at encoding time, once we know the encoding for “the” we can cache that information leading to further speed ups. But using spaces as a special character has drawbacks, namely:

- We can’t address as well languages that don’t use a space to separate words like Chinese (arguably German).

- We can’t encode frequently occurring multi words like “New York” or “European Union” or “black holes”

The second problem is especially bad when you consider examples where semantic is very different from the composing words like “Chicago Bulls” have nothing to do with bulls.

## **ε-BPE or model based BPE encoding**

The core idea is that instead of using frequency in the dataset to create the byte pairs, we can use the probability transition of the model to create the BPE. Let’s use some kind of transformer, GPT-2 for instance. The core idea of that model, is to predict the next token (in the BPE sense) given a fixed context size. But we can use the output probability of the model in order to create new tokens, not because they are frequent but because they are easy to predict. For instance in a book that contains a character "Sir Francis" that appears rarely, but there is only one character named "Sir …", the algorithm might learn quite easily that "Sir " is followed by "Francis" with great confidence, even if the occurence of the words is pretty low compared to common words like "the", "like" and "I".

So the core algorithm, will train a simple transformer on a dataset on regular bytes (at least at the start). Then, as the algorithm learns, some predictions will be above 1-ε. We can keep track of those and keep track of the last token we received, to check if we were correct.

Let's keep a hit map to see how successful our algorithm is. For instance, I predicted "Fo" will be followed by "gg" (Phileas Fogg is a character in Around the world in 80 days) with probability > 1-ε. I was correct in 14 cases, and got it wrong in 1 case (let's say it was classical "Fo" "g "). We were correct 14/15 times that's 93% accuracy. If we look at the fluctuation interval associated with that, we get [92.74-93.25%] range. If 92.74 > 1–ε we can conclude that our transition prediction is really very good, it's not a fluke of the model.

More generally, if we want 95% confidence when we upgrade this transition, we need to respect the following inequality : k / n - 1/sqrt(n) > 1-ε, where k is the number of successful predictions, n is the total number of predictions and ε the probability margin explained earlier.

This model is slightly different from byte pair encoding, but now we don’t suffer from the 2 problems mentioned above, we can get pretty long tokens if the dataset allows for it, and we can use Chinese or German as the space character does not play any special role.

## **Results**

Implementation can be found here. On the first run, we ran on a book [Around the world in 80 days](https://en.wikipedia.org/wiki/Around_the_World_in_Eighty_Days) by Jules Verne. It’s a very small dataset but the idea is to check that we can actually overcome BPE’s limitations. Here are a few telling tokens that were created while running on the dataset :

| Promotion # | Token created                          |
| ----------- | -------------------------------------- |
| 338         | "Mr. Fogg"                             |
| 357         | "Phileas Fogg"                         |
| 360         | "Passepartout"                         |
| 635         | "ir Franc" (Sir Francis)               |
| 781         | "It was"                               |
| 900         | '" asked' (contains a quote character) |

What is interesting, it that:

- We managed to create multi word tokens like “Phileas Fogg”

- Multi word tokens are a minority in terms of tokens created by the algorithm. Out of 421 tokens that contain a space character only 27 are multi word tokens like “New York”. The remaining 394 tokens contain an ending space, meaning our algorithm is learning word boundaries. It is reassuring because traditional BPE are usually hardcoding that information.

- Multi word tokens are name of characters in the book, which are occurring frequently, they are an entity by themselves (Fogg even has 2 tokens associated to him)

- 2 Multi word tokens are **not** specific to the book, “it was” is a pretty common 2 word token in English in descriptions, “(…) asked” is a very common continuation when we start a quote and end a sentence with a question mark. We can guess that “(…) said” would be a token further down the line, but it’s harder as there are probably a wider variety of verbs that can fit (said, replied, answered and so on…)

Here is a more complete comparison of standard BPE with ε-BPE, with the first 100 tokens generated, as you can see more tokens are dedicated to syntax in eBPE, which Standard BPE ignore gladly by splitting on newlines and spaces.

| Standard BPE | eBPE        |
| ------------ | ----------- |
| 'th'         | '\r\n'      |
| 'the '       | ', '        |
| 'an'         | 'd '        |
| 'in'         | 'Th'        |
| 'ou'         | 've'        |
| 'er'         | 'y '        |
| 'ed '        | '; '        |
| 'ar'         | 'f '        |
| 'hi'         | ',\r\n'     |
| 'on'         | '\r\n\r\n'  |
| 're'         | 'th'        |
| 'en'         | 'qu'        |
| 'and '       | 'the'       |
| 'of '        | ' '         |
| 'st'         | 'the '      |
| 'to '        | 'The'       |
| 'as '        | '\r\n'      |
| 'se'         | ', '        |
| 'ha'         | 'y '        |
| 'or'         | 'd '        |
| '.\r '       | 'Th'        |
| 'it'         | 've'        |
| 'he '        | '; '        |
| 'le'         | 'f '        |
| 'ing '       | ',\r\n'     |
| ',\r '       | ' '         |
| 'as'         | '\r\n'      |
| 'in '        | ', '        |
| 'at'         | 'd '        |
| 'at '        | 'y '        |
| 'ro'         | 'Th'        |
| 'er '        | 've'        |
| 'al'         | 'f '        |
| 'es'         | '; '        |
| 'on '        | ' '         |
| 'was '       | ',\r\n'     |
| 'no'         | 'th'        |
| 'his '       | '\r\n'      |
| 'ed'         | ', '        |
| 'ac'         | 'd '        |
| '"\r '       | 'y '        |
| 'ri'         | 'Th'        |
| 'be'         | 've'        |
| 'ly '        | 'f '        |
| 'om'         | '; '        |
| 'li'         | ' '         |
| 'en '        | ',\r\n'     |
| 'ti'         | 'th'        |
| 'og'         | '\r\n\r\n'  |
| 'ra'         | 'the'       |
| 'di'         | 'the '      |
| 'art'        | 'The'       |
| 'Fog'        | 'qu'        |
| 'the'        | 's '        |
| 'ma'         | 'The '      |
| 've '        | 'g '        |
| 'is '        | ',"'        |
| 'or '        | 'no'        |
| 'ld '        | 't '        |
| 'whi'        | 'th '       |
| 'il'         | 'o '        |
| 'ur'         | '?"'        |
| 's, '        | '\r\n\r\n"' |
| 'de'         | '," '       |
| 'wh'         | 'Mr'        |
| 'lo'         | 'e '        |
| 'ch '        | 'yo'        |
| 'ere '       | 'Yo'        |
| 'ith '       | 'ou'        |
| 'The '       | '. '        |
| 'am'         | 'nd '       |
| 'ent'        | 'h '        |
| 'un'         | 'n '        |
| 'gh'         | ';\r\n'     |
| 'with '      | 'og'        |
| 'an '        | 'you'       |
| 'oun'        | 'r '        |
| 'part'       | 'of '       |
| 'ver'        | 'to '       |
| 'si'         | 's F'       |
| 'had '       | 'Pa'        |
| 'not '       | 'as '       |
| 'ould '      | ''s '       |
| 'ing'        | '. F'       |
| 'out '       | 'is '       |
| 'el'         | 'ld '       |
| 'sa'         | 'ng '       |
| 'ce'         | 'at '       |
| 'that '      | 're'        |
| 'asse'       | 've '       |
| 'fi'         | 'gh'        |
| 'ol'         | 'ut '       |
| 'sh'         | 'll'        |
| 'r. '        | 'Pas'       |
| '."\r '      | 're '       |
| 'Passe'      | 'ed '       |
| 'Passepart'  | '. Fog'     |
| 'ut '        | 'ch '       |
| 'which '     | 'and '      |
| 'ay'         | 'ea'        |

I would love to check the tokenization of German or Chinese but I’m not a speaker of either language so it’s hard for me to analyze the results anyway. What’s for sure is that the technique is applicable.

I also tried the technique on different types of files like wav files or mp3 files, even jpeg images. Analysis is harder to do. Still some interesting notes, it took longer for the model to emit new tokens on the mp3 files than on the wav files. The mp3 file is encoded, therefore should have a lower entropy (meaning it’s harder to predict the next token) than the wav files so the model takes longer to actually get good at predicting. It’s probable (I haven’t checked) that we have to overfit the mp3 file and jpeg files before we can predict any meaningful content (except maybe the header part)

## **Future Work**

Many interesting ideas are still left to explore to continue exploring the idea of models creating their own tokenization. For now a limiting factor is the actual BPE encoding process that takes longer and longer as the model creates new tokens. That's because the encoding process is done in Python, so it's quite slow and can’t be precalculated as you would do with fixed BPE encodings. To give a sense of the slowdown, the training loop starts at ~11it/s on a GTX970 and finished at roughly 10s/it. That’s a 100x slowdown over the course of the training, with only 1k tokens in the end, far from the 50k used by GPT-2 for instance.

It’s going to be an actual requirement to train on larger and more representative datasets. Training on bigger datasets would help us understand how important are those multi word tokens and maybe what are those multi words. The token "(…) **asked**" was pretty surprising to me, I'm eager to see what else can be discovered.

The actual epsilon used was 40% which actually quite a big (value was chosen with trial and error, to get a small but not null rejection rate of new tokens, to add tokens as fast as possible but not making too many mistakes). That value probably has a sweet spot depending on the number of current tokens, after speeding up the process it would be interesting to look at the best value for epsilon as a function of the number of tokens.

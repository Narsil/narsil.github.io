---
layout: post
title: 'Creating a translation app'
author: nicolas
toc: true
description: How to create a custom clone of translate.google.com
categories: [ml, nlp, react]
---

> TL;DR Recently moved to the Netherlands, in order to avoid Googling translate everything, I did the next best thing to learning the language: I created a clone of translate.google.com

## Find a correct training loop

My first instinct was to check [Hugging Face](https://github.com/huggingface/transformers) as this repo contains solid implementations that I know are easy to change. However, in that particular instance, the example for translation does not start from scratch, and I wanted to check what multilingual translation could do here, as I'm using English, Dutch & French on translate.google.com (For food sometimes french is much better than english for me).

My second guess was [Fairseq](https://github.com/pytorch/fairseq) from facebook. In their example there is an actual example for multilingual German, French, English. Close enough for my needs. First things first, start to follow the example by the book. Most implementations out there are broken and won't work out of the box.

This time, it turned out particularly smooth. Clone the repo then follow the [instructions](https://github.com/pytorch/fairseq/tree/master/examples/translation#multilingual-translation)

```
# First install sacrebleu and sentencepiece
pip install sacrebleu sentencepiece

# Then download and preprocess the data
cd examples/translation/
bash prepare-iwslt17-multilingual.sh
cd ../..

# Binarize the de-en dataset
TEXT=examples/translation/iwslt17.de_fr.en.bpe16k
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en,$TEXT/valid2.bpe.de-en,$TEXT/valid3.bpe.de-en,$TEXT/valid4.bpe.de-en,$TEXT/valid5.bpe.de-en \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Binarize the fr-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train.bpe.fr-en \
    --validpref $TEXT/valid0.bpe.fr-en,$TEXT/valid1.bpe.fr-en,$TEXT/valid2.bpe.fr-en,$TEXT/valid3.bpe.fr-en,$TEXT/valid4.bpe.fr-en,$TEXT/valid5.bpe.fr-en \
    --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Train a multilingual transformer model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
mkdir -p checkpoints/multilingual_transformer
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --task multilingual_translation --lang-pairs de-en,fr-en \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir checkpoints/multilingual_transformer \
    --max-tokens 4000 \
    --update-freq 8

# Generate and score the test set with sacrebleu
SRC=de
sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src \
    | python scripts/spm_encode.py --model examples/translation/iwslt17.de_fr.en.bpe16k/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-en.${SRC}.bpe
cat iwslt17.test.${SRC}-en.${SRC}.bpe \
    | fairseq-interactive data-bin/iwslt17.de_fr.en.bpe16k/ \
      --task multilingual_translation --lang-pairs de-en,fr-en \
      --source-lang ${SRC} --target-lang en \
      --path checkpoints/multilingual_transformer/checkpoint_best.pt \
      --buffer-size 2000 --batch-size 128 \
      --beam 5 --remove-bpe=sentencepiece \
    > iwslt17.test.${SRC}-en.en.sys
```

## The data

While it's training, let's look at where I can get Dutch data. The IWSLT 2017 did not seem to have Dutch data [at first glance](https://wit3.fbk.eu/mt.php?release=2017-01-trnted) or [here](https://wit3.fbk.eu/mt.php?release=2017-01-trnmted). I also tried just mimicking the adress from facebook `prepare-iwslt17-multilingual.sh` (The address `https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz` so I simply tried if `https://wit3.fbk.eu/archive/2017-01-trnted/texts/nl/en/nl-en.tgz`). Turns out there aren't.
[Europarl](https://www.statmt.org/europarl/) seemed like a good bet but looking at the data, the langage seems pretty formatted and not very dialogue like. That might explain why it does not seem to be used that often.
Looking back at IWSLT 2017 finally found the [Dutch data](https://wit3.fbk.eu/mt.php?release=2017-01-mted-test) and the [training data](https://wit3.fbk.eu/mt.php?release=2017-01-trnmted). Is it me, or are competitions websites really hard to read ?

## The actual training loop

Ok so let's reuse the training loop from the german file, so we just need to copy the dutch files in the same layout as the german ones, edit all the scripts and command lines to edit everything. I had to multiply the test files, someone Facebook has tst2011, tst2012 tst2013, tst2014, tst2015 for the german data, which does not seem to exist on the competition website... So here instead of trying to figure out where the information was, I simply copy-pasted the tst2010 file into dummy versions for tst2011...tst2015 (oh yeah simply omitting them will make bash scripts fail because file alignement is a requirement !, and I don't want to spend more than 5mn editing a bash script).

Now with our edited bash script:

```
cd examples/translation/
bash prepare-iwslt17-multilingual_nl.sh
cd ../..
```

Preprocess dutch data:

```
TEXT=examples/translation/iwslt17.nl.en.bpe16k
fairseq-preprocess --source-lang nl --target-lang en \
    --trainpref $TEXT/train.bpe.nl-en \
    --validpref $TEXT/valid0.bpe.nl-en,$TEXT/valid1.bpe.nl-en,$TEXT/valid2.bpe.nl-en,$TEXT/valid3.bpe.nl-en,$TEXT/valid4.bpe.nl-en,$TEXT/valid5.bpe.nl-en \
    --destdir data-bin/iwslt17.nl_fr.en.bpe16k \
    --workers 10
```

Now let's preprocess french data

```
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train.bpe.fr-en \
    --validpref $TEXT/valid0.bpe.fr-en,$TEXT/valid1.bpe.fr-en,$TEXT/valid2.bpe.fr-en,$TEXT/valid3.bpe.fr-en,$TEXT/valid4.bpe.fr-en,$TEXT/valid5.bpe.fr-en \
    --tgtdict data-bin/iwslt17.nl_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.nl_fr.en.bpe16k \
    --workers 10
```

Overall, pretty simple task, just a bit bothering to hit all the various walls.

Now that we preformatted the dutch data, we can run the training loop on our own data !

```
mkdir -p checkpoints/multilingual_transformer_nl
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.nl_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --task multilingual_translation --lang-pairs nl-en,fr-en \
    # Don't change the arch !\
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    # Change the checkpoint \
    --save-dir checkpoints/multilingual_transformer_nl \
    --max-tokens 4000 \
    --update-freq 8
```

## Checking the final result

So now we have a model `checkpoints/multilingual_transformer_nl/checkpoint_best.pt`, let's run it !

```
# Generate and score the test set with sacrebleu
SRC=nl
sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src \
    | python scripts/spm_encode.py --model examples/translation/iwslt17.nl_fr.en.bpe16k/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-en.${SRC}.bpe
cat iwslt17.test.${SRC}-en.${SRC}.bpe \
    | fairseq-interactive data-bin/iwslt17.nl_fr.en.bpe16k/ \
      --task multilingual_translation --lang-pairs de-en,fr-en \
      --source-lang ${SRC} --target-lang en \
      --path checkpoints/multilingual_transformer_nl/checkpoint_best.pt \
      --buffer-size 2000 --batch-size 128 \
      --beam 5 --remove-bpe=sentencepiece \
    > iwslt17.test.${SRC}-en.en.sys
```

But woops...`sacreBLEU: No such language pair "nl-en" sacreBLEU: Available language pairs for test set "iwslt17": en-fr, fr-en, en-de, de-en, en-zh, zh-en, en-ar, ar-en, en-ja, ja-en, en-ko, ko-en`

So it looks like we're going to need to pipe some of our own data into this pipeline, we can just use the validation set we used to train

```
cat examples/translation/iwslt17.nl_fr.en.bpe16k/valid0.bpe.nl-en.nl |
python scripts/spm_encode.py --model examples/translation/iwslt17.nl_fr.en.bpe16k/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-en.${SRC}.bpe
```

There we go we have encoded with our multilingual BPE tokenizer our valid dataset. We can now run our translating command

```
cat iwslt17.test.${SRC}-en.${SRC}.bpe     | fairseq-interactive data-bin/iwslt17.nl_fr.en.bpe16k/       --task multilingual_translation --lang-pairs nl-en,fr-en       --source-lang ${SRC} --target-lang en       --path checkpoints/multilingual_transformer_nl/checkpoint_best.pt       --buffer-size 2000 --batch-size 128       --beam 5 --remove-bpe=sentencepiece
```

Here are some outputs (not cherry picked):

```rust
S-999   Iedereen heeft een vissenkom nodig.
H-999   -1.0272072553634644     Everybody needs a fishing ticket.
D-999   -1.0272072553634644     Everybody needs a fishing ticket.
P-999   -1.5687 -0.2169 -0.2363 -2.0637 -2.6527 -0.2981 -0.1540
```

```rust
S-998   Het leidt tot meer verlamming en minder tevredenheid.
H-998   -0.32848915457725525    It leads to more paralysis and less satisfaction.
D-998   -0.32848915457725525    It leads to more paralysis and less satisfaction.
P-998   -0.9783 -0.3836 -0.1854 -0.8328 -0.1779 -0.0163 -0.3334 -0.3619 -0.2152 -0.0450 -0.2831 -0.1289
```

```rust
S-987   Ze maken ons leven minder waard.
H-987   -0.5473383665084839     They make our lives worth less.
D-987   -0.5473383665084839     They make our lives worth less.
```

Seems good enough for now.

## Productizing

### Flask server

Ok, in order to productionize, initially I wanted to move away from fairseq, but a lot of logic is actually tied to fairseq-interative (beam search, loading all the args, ensembling the model, source language selection and so on). It's definitely possible to move out of it, but it felt like a few days job, so much more than I was willing to invest in this particular approach.

So the idea is to have a flask server sitting in front of the model, call the appropriate encoding with spm_encode, pass it to fairseq interactive, and output the D-XXX line back to the caller.

We're going to containerize it and deploy to Kubernetes (it just happens I have a kubernetes cluster running, so less problems with deploying on it). I considered using ONNX-js (or TFlite) to deploy directly on the browser which saves a lot of headaches on deployment and keeping the service running in the long run (Like I did for the [glasses](https://narsil.github.io/assets/face/) project). Here the main problem is the size of the model (600Mo). I could go back and try to optimize but that's a pretty big model, it's going to be hard to make it come to a comfortable level for browser-only mode (Again just too much work for what I have in mind here).

So let's get started from the Flask's [hello world](https://flask.palletsprojects.com/en/1.1.x/quickstart/)

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
```

Let's edit it a bit to include our translate function.

```python
from flask import Flask
app = Flask(__name__)

def translate(text):
    # TODO later
    return "This is a translation !"

@app.route('/', methods=["POST"])
def hello():
    text = request.form["input"]
    print(f"IN {text}")
    output = translate(text)
    print(f"OUT {output}")
    result = json.dumps({"en": output})
    return result
```

We can run our example and check it's running with curl

```
$ curl -d input="Ik heft een appel." http://localhost:5000/`
{"en": "This is a translation !"}
```

### Implementing the translate function.

Ok this is where we are super tied to fairseq-interactive code, I had to dig into the source code, copy most of it, and mainly split `Model loading` code from `Model running` code. For that I used a lot of globals as the original code does not separate these two concerns (tidying this will be a later goal if it every comes to that).

The final implementation is quite verbose but available [here](https://github.com/Narsil/translate/blob/master/server/translate.py).

One good point about this implementation is that we load the model early, so that it's available right away when the server comes up (but it does take some time to come up).
A negative point, is that because it's loaded eagerly it's going to make forking a nightmare and basically preventing us from using wsgi efficiently which is the [recommended way of deploying Flask](https://flask.palletsprojects.com/en/1.1.x/deploying/). It's fine for now, it's a personnal project after all, to get more stable deployment I would try to remove python from the equation of the web part if possible, it's really slow and hard to work with on webservers because of the forking/threading nightmare in Python.

So know our backend can really translate !

```
$ curl -d input="Ik heft een appel." http://localhost:5000/`
{"en": "I have an apple."}
```

Before moving that to the cloud, let's build a nice interface in front of it

### React front

Ok so we're going to use React with Typescript. React because we're going JS anyway to get the translation without clicking a button with a form like html. It's also more convenient to use Material-UI which I find helps make a website nice from scratch (and I'm tired of Bootstrap). Typescript because it's just saner than VanillaJS (it won't make much of a difference here).

```
yarn create react-app app --template typescript
cd app
yarn add @material-ui/core
```

Let's edit our App.tsx to use Material-UI and get the initial layout looking like [translate.google.com](translate.google.com).

```tsx
import React from 'react';
import {makeStyles} from '@material-ui/core/styles';
import TextField from '@material-ui/core/TextField';
import Card from '@material-ui/core/Card';
import Grid from '@material-ui/core/Grid';

const useStyles = makeStyles(theme => ({
  app: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
  },
}));

function App() {
  const classes = useStyles();

  return (
    <div className={classes.app}>
      <Card>
        <form>
          <Grid container>
            <Grid item xs={12} md={6}>
              <TextField
                id="standard-basic"
                label="Dutch"
                multiline
                autoFocus
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField id="standard-basic" label="English" multiline />
            </Grid>
          </Grid>
        </form>
      </Card>
    </div>
  );
}
export default App;
```

Here is the result : ![](https://i.imgur.com/ZszCVQU.png)

Now let's look at the logic (simplified):

```typescript
type Timeout = ReturnType<typeof setTimeout>;

const [text, setText] = useState('');
const [time, setTime] = useState<Timeout | null>(null);
const url = 'http://localhost:5000';

const translate = (text: string) => {
  if (text === '') {
    setText('');
    return;
  }
  const form = new FormData();
  form.append('input', text);
  fetch(url, {
    method: 'POST',
    body: form,
  }).then(response => {
    response.json().then(json => {
      console.log(json);
      setText(json['en']);
    });
  });
};
```

Then call it on the `onChange` attribute of our Dutch field.

```typescript
onChange={event => {
    // We use a timeout handler to prevent very fast keystrokes
    // from spamming our server.
    if (time !== null) {
        clearTimeout(time);
    }
    const text = event.target.value;
    const timeout = setTimeout(() => {
        translate(text);
    }, 500);
    setTime(timeout);
}}
```

There we have it:

![](https://i.imgur.com/EYZ0EWR.gif)

### Let's dockerize !

As I mentionned loading the whole model in the flask app is going to hinder a lot the wsgi process forking. I did try it, try to come up with easy fixes, but ultimately found that keeping the development server was just easier.

Ok so we're going to need a python docker image, install pytorch, fairseq, and flask to our image (actually we need flask_cors too to make sure we can call from any website as it's an API.)

As it turns out, fairseq 0.9 had a bug in the training loop and I was using master from a few month ago, and I needed to work with that specific version since there had been breaking changes since in master. That gives us the following `requirements.txt`

```
torch
flask
flask_cors
-e git://github.com/pytorch/fairseq.git@7a6519f84fed06947bbf161c7b66c9099bc4ce53#egg=fairseq
sentencepiece
```

Now our Docker file, is going to get the python dependencies, copy all the local files (including model and tokenizer file) and run the flask server. That gives us :

```Dockerfile
FROM python:3.7-slim
RUN pip install -U pip
RUN apt-get update && apt-get install -y git build-essential # Required for building fairseq from source.
COPY server/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "translate.py"]
```

Let's build and check that it works:

```
docker build -t translate:latest .
docker run -p 5000:5000 translate:latest
# Now check with curl that we can still hit the docker and get a correct answer
curl -d input="Ik heft een appel." http://localhost:5000/`
# {"en": "This is a translation !"}
```

### Kubernetes cluster

Okay the following part will be pretty specific to my setup. I use a kubernetes cluster on GCP with ingress. I'm going to skip updating the SSL certificate.

Let's start with pushing the image to GCP:

```
docker tag translate:latest gcr.io/myproject-XXXXXX/translate:1.0
docker push gcr.io/myproject-XXXXXX/translate:1.0
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

```

Here are the (edited for brevity&security) service files I used:

```yaml
#deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: translate-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: translate
  template:
    metadata:
      labels:
        app: translate
    spec:
      containers:
        - name: translate
          image: gcr.io/myproject-XXXXX/translate:1.0
          ports:
            - containerPort: 5000
```

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: translate-service
spec:
  type: NodePort
  selector:
    app: translate
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
```

```yaml
#ingress.yaml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-front
  annotations:
    kubernetes.io/ingress.global-static-ip-name: address-cluster
    networking.gke.io/managed-certificates: ottomate-certificate-new
spec:
  rules:
    - host: translate.ottomate.app
      http:
        paths:
          - path: /*
            backend:
              serviceName: translate-service
              servicePort: 80
```

Hopefully within a few minutes you have your pod running and you can hit your live own server with the API.

You just need to update your react App to point the the correct URL and boom your done, your very own translate server app.

### What could/should be done next.

#### For the model:

- Add more data to the original training set, some words are missing, translation can become funky on some real world sentences I give the machine (Dutch companies tend to send very verbose emails)
- Add some data augmentation in the pool as the current translation is very brittle to errors. Using Sentence piece algorihm with sampling instead of BPE could be used, some typo generator, word inversions to name a few. Training some error detection algorithm on top or using ready made ones could help (translate.google.com has some spellfixing magic applied before it seems.)
- Making it smaller to make it portable to tflite, mobile phone for offline mode and so on (it's a pretty big workload to make it work though)

#### For the backend:

- Battle testing the backend should be the first thing to do to check failure modes and fix naive DOS attacks.
- Something like [TorchServe](https://github.com/pytorch/serve) seems like what we want for the model part. Never used it so far, but it seems to solve some problems encountered here and would make iterations faster on various models (also swapping out models).
- On the other spectrum I could go for tighter control. Removing the fairseq-interative clutter would be my first move. If I can go pytorch barebones, then using Rust, with Hugging Face's [tokenizers](https://github.com/huggingface/tokenizers) library would probably make inference faster and deployment easier. It would of course make iteration much slower so I would do that only when the model is very stable. It could make mobile offline possible (with a very large app data, ~1Go, but doable.)

#### For the frontend:

- Working a bit more on the mobile part of the design which is a bit broken at the moment.
- Maybe add buttons to switch languages easily, switch language sides (although I mostly use Dutch->English and Dutch->French)
- Add a react-native app so that I can translate from my phone. (Without offline mode)

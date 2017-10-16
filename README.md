# deepspeech-playground

This repo is a fork of [Baidu's DeepSpeech](https://github.com/baidu-research/ba-dls-deepspeech).  Unlike Baidu's repo:

- It works with both Tensorflow and Theano
- It has helpers for better training by training against auto-generated phonograms
- Training by Theano can be much faster, since CTC calculation may be done by GPU


## Training

If you want train by Theano you'll need Theano>=0.10 since It has [bindings](http://deeplearning.net/software/theano_versions/dev/library/tensor/nnet/ctc.html) for Baidu's CTC.

### Using Phonogram

`HalfPhonemeModelWrapper` class in `model_wrp` module implements training of a model with half of RNN layers trained for Phonorgrams and rest of them for actual output text. To generate Phonograms, [Logios](https://github.com/skerit/cmusphinx/tree/master/logios) tool of CMU Sphinx can be used.  Sphinx Phonogram symbols are called [Arpabets](http://www.speech.cs.cmu.edu/cgi-bin/cmudict).  To generate Arpabets from Baidu's DeepSpeech [description files](https://github.com/baidu-research/ba-dls-deepspeech#data) you can:
```
$ cat train_corpus.json | sed -e 's/.*"text": "\([^"]*\)".*/\1/' > train_corpus.txt
# make_pronunciation.pl script is provided by logios
# https://github.com/skerit/cmusphinx/tree/master/logios/Tools/MakeDict
$ perl ./make_pronunciation.pl -tools ../ -dictdir .  -words prons/train_corpus.txt -dict prons/train_corpus.dict
$ python create_arpabet_json.py train_corpus.json train_corpus.dict train_corpus.arpadesc
```

### Choose backend

Select Keras backend by environment variable `KERAS_BACKEND` to `theano` or `tensorflow`.

### Train!
Make a train routine, a function like this:

```
def train_sample_half_phoneme(datagen, save_dir, epochs, sortagrad,
                              start_weights=False, mb_size=60):
    model_wrp = HalfPhonemeModelWrapper()
    model = model_wrp.compile(nodes=1000, conv_context=5, recur_layers=5)
    logger.info('model :\n%s' % (model.to_yaml(),))

    if start_weights:
        model.load_weights(start_weights)

    train_fn, test_fn = (model_wrp.compile_train_fn(1e-4),
                         model_wrp.compile_test_fn())
    trainer = Trainer(model, train_fn, test_fn, on_text=True, on_phoneme=True)
    trainer.run(datagen, save_dir, epochs=epochs, do_sortagrad=sortagrad,
                mb_size=mb_size, stateful=False)
    return trainer, model_wrp
```
And call it in from `main()` of `train.py`. Training can be done by:
```
$ KERAS_BACKEND="tensorflow" python train.py descs/small.arpadesc descs/test-clean.arpadesc models/test --epochs 20 --use-arpabets --sortagrad 1```
```

## Evaluation

`visualize.py` will give you a semi-shell for testing your model by giving it input files. There is also [models-evaluation notebook](models-evaluation.ipynb), though it may look too dirty..

## Pre-trained models

These models are trained for about three days by LibriSpeech corpus on a GTX 1080 Ti GPU:

- A five layers unidirectional RNN model trained by LibriSpeech using Theano:  [mega](https://mega.nz/#!ZTIjXQgA!HK1vCRxYC1VyzJ_8LCwwcTrNH9aF7l-H8TYf7eE1v6g)
- A five layers unidirectional RNN model trained by LibriSpeech using Tensorflow: [mega](https://mega.nz/#!APR1iRjT!pgJcnEWLTHzJ4m9dQXA_2gvrJxa_h9uwEHc6Sxwreow)

Validation WER of these models on `test-clean` is about %5 an It's about %15 on `test-other`.

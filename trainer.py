import os
import logging
import numpy as np

from utils import conv_chain_output_length, word_error_rate, save_model

logger = logging.getLogger(__name__)


def _last_of_list_or_none(l):
    return None if len(l) == 0 else l[-1]


class Trainer(object):
    """
    Training and validation routines

    Properties:
        best_cost (flaot)
        last_cost (float)
        best_val_cost (float)
        last_val_cost (float)
        wers (list(float))
        val_wers (list(float))
    """
    def __init__(self, model, train_fn, val_fn, on_text=True, on_phoneme=False):
        self.model = model
        self.train_fn = train_fn
        self.val_fn = val_fn
        self.on_text = on_text
        self.on_phoneme = on_phoneme
        self.wers, self.text_costs, self.phoneme_costs = [], [], []
        self.val_wers, self.val_text_costs, self.val_phoneme_costs = [], [], []
        self.best_cost = np.iinfo(np.int32).max
        self.best_val_cost = np.iinfo(np.int32).max
        if not (on_text or on_phoneme):
            raise ValueError("Model should train against at least text or "
                             "phoneme")

    @property
    def last_text_cost(self):
        return _last_of_list_or_none(self.text_costs)

    @property
    def last_phoneme_cost(self):
        return _last_of_list_or_none(self.phoneme_costs)

    @property
    def last_val_text_cost(self):
        return _last_of_list_or_none(self.val_text_costs)

    @property
    def last_val_phoneme_cost(self):
        return _last_of_list_or_none(self.val_phoneme_costs)

    @property
    def last_wer(self):
        return _last_of_list_or_none(self.wers)

    @property
    def last_val_wer(self):
        return _last_of_list_or_none(self.val_wers)

    @property
    def last_cost(self):
        """ Cost of last minibatch on train """
        if self.on_text:
            return self.last_text_cost
        if self.on_phoneme:
            return self.last_phoneme_cost

    @property
    def last_val_cost(self):
        """ Last cost on whole validation set """
        if self.on_text:
            return self.last_val_text_cost
        if self.on_phoneme:
            return self.last_val_phoneme_cost

    @property
    def best_cost(self):
        """ Best cost among minibatchs of training set """
        if self.on_text:
            return self.best_text_cost
        if self.on_phoneme:
            return self.best_phoneme_cost

    @best_cost.setter
    def best_cost(self, val):
        if self.on_text:
            self.best_text_cost = val
        elif self.on_phoneme:
            self.best_phoneme_cost = val

    @property
    def best_val_cost(self):
        """ Best cost on whole validation set so far """
        if self.on_text:
            return self.best_text_val_cost
        if self.on_phoneme:
            return self.best_phoneme_val_cost

    @best_val_cost.setter
    def best_val_cost(self, val):
        if self.on_text:
            self.best_text_val_cost = val
        elif self.on_phoneme:
            self.best_phoneme_val_cost = val

    def run(self, datagen, save_dir, epochs=10, mb_size=16, do_sortagrad=False,
            stateful=False, save_best_weights=False, save_best_val_weights=True,
            iters_to_valid=100, iters_to_checkout=500):
        """ Run trainig loop
            Args:
                datagen (DataGenerator)
                save_dir (str): directory path that will contain the model
                epochs (int): number of epochs
                mb_size (int): mini-batch size
                do_sortagrad (bool): sort dataset by duration on first epoch
                stateful (bool): is model stateful or not
                save_best_weights (bool): save weights whenever cost over
                    training mini-batch reduced
                save_best_val_weights (bool): save weights whenever cost over
                    validation set reduced
                iters_to_valid (int): after this amount of iterations validate
                    model by whole validation set
                iters_to_checkout (int): after this amount of iterations save
                    model
        """
        logger.info("Training model..")
        iters = 0
        for e in range(epochs):
            if not isinstance(do_sortagrad, bool):
                sortagrad = e < do_sortagrad
                shuffle = not sortagrad
            elif do_sortagrad:
                shuffle = False
                sortagrad = True
            else:
                shuffle = True
                sortagrad = False

            train_iter = datagen.iterate_train(mb_size, shuffle=shuffle,
                                               sort_by_duration=sortagrad)
            for i, batch in enumerate(train_iter):
                if stateful and batch['x'].shape[0] != mb_size:
                    break
                self.train_minibatch(batch, i % 10 == 0)
                if i % 10 == 0:
                    logger.info("Epoch: {} Iteration: {}({}) TextLoss: {}"
                                " PhonemeLoss: {} WER: {}"
                                .format(e, i, iters, self.last_text_cost,
                                        self.last_phoneme_cost,
                                        self.last_wer))
                iters += 1
                if save_best_weights and self.best_cost < self.last_cost:
                    self.save_weights(save_dir, 'best-weights.h5')
                if iters_to_valid is not None and iters % iters_to_valid == 0:
                    self.validate(datagen, mb_size, stateful,
                                  save_best_val_weights, save_dir)
                if i and i % iters_to_checkout == 0:
                    self.save_model(save_dir, iters)
            if iters_to_valid is not None and iters % iters_to_valid != 0:
                self.validate(datagen, mb_size, stateful, save_best_val_weights,
                              save_dir)
            if i % iters_to_checkout != 0:
                self.save_model(save_dir, iters)

    def train_minibatch(self, batch, compute_wer=False):
        inputs = batch['x']
        input_lengths = batch['input_lengths']
        ctc_input_lens = self.ctc_input_length(input_lengths)
        if self.on_text and self.on_phoneme:
            _, ctc_phoneme, pred_texts, ctc_text = self.train_fn([
                inputs, ctc_input_lens, batch['phonemes'],
                batch['phoneme_lengths'], batch['y'], batch['label_lengths'],
                True])
        elif self.on_text:
            pred_texts, ctc_text = self.train_fn([inputs, ctc_input_lens,
                                                  batch['y'],
                                                  batch['label_lengths'], True])
        elif self.on_phoneme:
            _, ctc_phoneme = self.train_fn([inputs, ctc_input_lens,
                                            batch['phonemes'],
                                            batch['phoneme_lengths'],
                                            True])
        if self.on_text:
            if compute_wer:
                wer = word_error_rate(batch['texts'], pred_texts).mean()
                self.wers.append(wer)
            self.text_costs.append(ctc_text)
        if self.on_phoneme:
            self.phoneme_costs.append(ctc_phoneme)

    def validate(self, datagen, mb_size, stateful, save_best_weights, save_dir):
        text_avg_cost, phoneme_avg_cost = 0.0, 0.0
        total_wers = []
        i = 0
        for batch in datagen.iterate_validation(mb_size):
            if stateful and batch['x'].shape[0] != mb_size:
                break
            text_cost, phoneme_cost, wers = self.validate_minibatch(batch)
            if self.on_text:
                text_avg_cost += text_cost
                total_wers.append(wers)
            if self.on_phoneme:
                phoneme_avg_cost += phoneme_cost
            i += 1
        if i != 0:
            text_avg_cost /= i
            phoneme_avg_cost /= i
        if self.on_text:
            self.val_wers.append(np.concatenate(total_wers).mean())
            self.val_text_costs.append(text_avg_cost)
        if self.on_phoneme:
            self.val_phoneme_costs.append(phoneme_avg_cost)
        logger.info("Validation TextLoss: {} Validation PhonemeLoss: {} "
                    "Validation WER: {}".format(self.last_val_text_cost,
                                                self.last_val_phoneme_cost,
                                                self.last_val_wer))
        if save_best_weights and self.last_val_cost < self.best_val_cost:
            self.best_val_cost = self.last_val_cost
            self.save_weights(save_dir, 'best-val-weights.h5')

    def validate_minibatch(self, batch):
        inputs = batch['x']
        input_lengths = batch['input_lengths']
        ctc_input_lens = self.ctc_input_length(input_lengths)
        text_ctc, phoneme_ctc, wers = None, None, None
        if self.on_text and self.on_phoneme:
            _, phoneme_ctc, pred_text, text_ctc = self.val_fn([
                inputs, ctc_input_lens, batch['phonemes'],
                batch['phoneme_lengths'], batch['y'], batch['label_lengths'],
                True])
        elif self.on_text:
            pred_text, text_ctc = self.val_fn([
                inputs, ctc_input_lens, batch['y'], batch['label_lengths'],
                True])
        elif self.on_phoneme:
            _, phoneme_ctc = self.val_fn([
                inputs, ctc_input_lens, batch['phonemes'],
                batch['phoneme_lengths'],  True
            ])

        if self.on_text:
            wers = word_error_rate(batch['texts'], pred_text)

        return text_ctc, phoneme_ctc, wers

    def ctc_input_length(self, input_lengths):
        import keras.layers
        conv_class = (getattr(keras.layers, 'Conv1D', None) or
                      keras.layers.Convolution1D)
        conv_lays = [l for l in self.model.layers if isinstance(l, conv_class)]
        return [conv_chain_output_length(l, conv_lays) for l in input_lengths]

    def save_weights(self, save_dir, filename):
        self.model.save_weights(os.path.join(save_dir, filename),
                                overwrite=True)

    def save_model(self, save_dir, index):
        save_model(save_dir, self.model, self.text_costs, self.val_text_costs,
                   wer=self.wers, val_wer=self.val_wers,
                   phoneme=self.phoneme_costs,
                   val_phoneme=self.val_phoneme_costs, index=index)

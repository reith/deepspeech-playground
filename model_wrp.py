"""
Model generator based on high level properties
"""

import logging
from utils import for_tf_or_th

import keras
import keras.backend as K
from keras.layers import (BatchNormalization, Dense, Input, GRU, concatenate,
                          TimeDistributed, Dropout)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

if K.backend() == 'theano':
    import theano.gpuarray.ctc as ctc_th
    logging.info('using theano.gpuarray.ctc')

k2 = keras.__version__[0] == '2'

if k2:
    from keras.layers import Conv1D

    def batch_norm_compat(self, mode=0, **kwargs):
        if mode != 0:
            logger.warn('ignoring unsuported batchnorm mode of {} on keras 2'
                        .format(mode))
        self._v2_init(**kwargs)
    BatchNormalization._v2_init = BatchNormalization.__init__
    BatchNormalization.__init__ = batch_norm_compat
else:
    from keras.layers import Convolution1D

    def v1_compat(self, *args,  **kwargs):
        v1_dict = {}
        for k, v in kwargs.items():
            if k == 'padding':
                v1key = 'border_mode'
            elif k == 'strides':
                v1key = 'subsample_length'
            elif k == 'kernel_initializer':
                v1key = 'init'
            else:
                v1key = k
            v1_dict[v1key] = v
        self._v1_init(*args, **v1_dict)

    Conv1D = Convolution1D
    Conv1D._v1_init = Conv1D.__init__
    Conv1D.__init__ = v1_compat

    Dense._v1_init = Dense.__init__
    Dense.__init__ = v1_compat

    GRU._v1_init = GRU.__init__
    GRU.__init__ = v1_compat


logger = logging.getLogger(__name__)


def duration_cost(y, y_pred):
    """" A Loss function for duration costs """
    return (y - y_pred)**2


def model_output_dim(out_type):
    """ Return output dimention of model based on output type
    Args:
        out_type: string either 'text' or 'arpabet'
    """
    if out_type == 'text':
        from char_map import index_map
        return len(index_map) + 1
    if out_type == 'arpabet':
        from arpabets import index_map
        return len(index_map) + 1
    raise ValueError


class ModelWrapper(object):

    def __init__(self, outputs='text', stateful=False):
        if outputs == 'text':
            self.output_dim = model_output_dim('text')
        elif outputs == 'arpabet':
            self.output_dim = model_output_dim('arpabet')
        elif isinstance(outputs, list) and (sorted(outputs) ==
                                            ['arpabet', 'text']):
            self.vocab_dim = model_output_dim('text')
            self.phono_dim = model_output_dim('arpabet')
        else:
            raise ValueError
        self.outputs = outputs
        self.stateful = stateful
        self.branch_vars = {}
        self.model = None

    @staticmethod
    def plug_model(old):
        if not isinstance(old, ModelWrapper):
            raise ValueError

        new = ModelWrapper(old.outputs, stateful=old.stateful)
        for attr in ('model', '_branch_labels', 'branch_vars', '_ctc_in_lens',
                     'branch_outputs', 'acoustic_input'):
            setattr(new, attr, getattr(old, attr))
        return new

    @property
    def branch_labels(self):
        if getattr(self, '_branch_labels', None) is None:
            d = dict()
            for bname in self.branch_vars.keys():
                d[bname] = (
                    K.placeholder(ndim=2, dtype='int32'),
                    K.placeholder(ndim=1, dtype='int32')
                )
            self._branch_labels = d
        return self._branch_labels

    @property
    def ctc_in_lens(self):
        if getattr(self, '_ctc_in_lens', None) is None:
            self._ctc_in_lens = K.placeholder(ndim=1, dtype='int32')
        return self._ctc_in_lens

    def compile_train_fn(self, learning_rate=2e-4):
        """ Build the CTC training routine for speech models.
        Args:
            learning_rate (float)
        Returns:
            train_fn (theano.function): Function that takes in acoustic inputs,
                and updates the model. Returns network outputs and ctc cost
        """
        logger.info("Building train_fn")
        f_inputs = [self.acoustic_input, self.ctc_in_lens]
        f_outputs = []
        f_updates = []
        for branch in self.branch_outputs:
            labels, label_lens = self.branch_labels[branch.name]
            f_inputs.append(labels)
            f_inputs.append(label_lens)

            if K.backend() == 'tensorflow':
                network_output = branch.output
                ctc_cost = K.mean(K.ctc_batch_cost(labels, network_output,
                                                   self.ctc_in_lens,
                                                   label_lens))
            else:
                network_output = branch.output.dimshuffle((1, 0, 2))
                ctc_cost = ctc_th.gpu_ctc(network_output, labels,
                                          self.ctc_in_lens).mean()

            f_outputs.extend([network_output, ctc_cost])
            trainable_vars = self.branch_vars[branch.name]
            optmz = Adam(lr=learning_rate, clipnorm=100)
            f_updates.extend(optmz.get_updates(trainable_vars, [], ctc_cost))

        f_inputs.append(K.learning_phase())
        self.train_fn = K.function(f_inputs, f_outputs, f_updates)
        return self.train_fn

    def compile_test_fn(self):
        """ Build a testing routine for speech models.
        Returns:
            val_fn (theano.function): Function that takes in acoustic inputs,
                and calculates the loss. Returns network outputs and ctc cost
        """
        logger.info("Building val_fn")
        f_inputs = [self.acoustic_input, self.ctc_in_lens]
        f_outputs = []
        for branch in self.branch_outputs:
            labels, label_lens = self.branch_labels[branch.name]

            if K.backend() == 'tensorflow':
                network_output = branch.output
                ctc_cost = K.mean(K.ctc_batch_cost(labels, network_output,
                                                   self.ctc_in_lens,
                                                   label_lens))
            else:
                network_output = branch.output.dimshuffle((1, 0, 2))
                ctc_cost = ctc_th.gpu_ctc(network_output, labels,
                                          self.ctc_in_lens).mean()

            f_inputs.extend([labels, label_lens])
            f_outputs.extend([network_output, ctc_cost])
        f_inputs.append(K.learning_phase())

        self.val_fn = K.function(f_inputs, f_outputs)
        return self.val_fn

    def compile_output_fn(self):
        """ Build a function that simply calculates the output of a model
        Returns:
            output_fn (theano.function): Function that takes in acoustic inputs,
                and returns network outputs
        """
        logger.info("Bulding output_fn")
        if self.outputs in ['text', 'arpabet']:
            output_idx = 0
        elif self.outputs == ['arpabet', 'text']:
            output_idx = 1
        else:
            raise ValueError

        output = self.model.outputs[output_idx]
        if K.backend() == 'theano':
            output = output.dimshuffle((1, 0, 2))

        output_fn = K.function([self.acoustic_input, K.learning_phase()],
                               [output])
        return output_fn


class GruModelWrapper(ModelWrapper):
    """ Recurrent network (CTC) for speech with GRU units """

    def compile(self, input_dim=161, recur_layers=3, nodes=1024,
                conv_context=11, conv_border_mode='valid', conv_stride=2,
                activation='relu', lirelu_alpha=.3, dropout=False,
                initialization='glorot_uniform', batch_norm=True,
                stateful=False, mb_size=None):
        logger.info("Building gru model")
        assert self.model is None

        leaky_relu = False
        if activation == 'lirelu':
            activation = 'linear'
            leaky_relu = True

        if stateful:
            if mb_size is None:
                raise ValueError("Stateful GRU layer needs to know batch size")
            acoustic_input = Input(batch_shape=(mb_size, None, input_dim),
                                   name='acoustic_input')
        else:
            acoustic_input = Input(shape=(None, input_dim),
                                   name='acoustic_input')

        # Setup the network
        conv_1d = Conv1D(nodes, conv_context, name='conv_1d',
                         padding=conv_border_mode, strides=conv_stride,
                         kernel_initializer=initialization,
                         activation=activation)(acoustic_input)

        if batch_norm:
            output = BatchNormalization(name='bn_conv_1d', mode=2)(conv_1d)
        else:
            output = conv_1d

        if leaky_relu:
            output = LeakyReLU(alpha=lirelu_alpha)(output)

        if dropout:
            output = Dropout(dropout)(output)

        for r in range(recur_layers):
            output = GRU(nodes, name='rnn_{}'.format(r + 1),
                         kernel_initializer=initialization, stateful=stateful,
                         return_sequences=True, activation=activation)(output)

            if batch_norm:
                bn_layer = BatchNormalization(name='bn_rnn_{}'.format(r + 1),
                                              mode=2)
                output = bn_layer(output)

            if leaky_relu:
                output = LeakyReLU(alpha=lirelu_alpha)(output)

        output_branch = TimeDistributed(Dense(
            self.output_dim, name='text_dense', init=initialization,
            activation=for_tf_or_th('softmax', 'linear')
        ), name=self.outputs)
        network_output = output_branch(output)

        self.model = Model(input=acoustic_input, output=[network_output])
        self.branch_outputs = [output_branch]
        self.branch_vars[output_branch.name] = self.model.trainable_weights
        self.acoustic_input = self.model.inputs[0]
        return self.model


class HalfPhonemeModelWrapper(ModelWrapper):

    def __init__(self, *args, **kwargs):
        super(HalfPhonemeModelWrapper, self).__init__(['arpabet', 'text'],
                                                      *args, **kwargs)

    def compile(self, input_dim=161, recur_layers=3, nodes=1024,
                conv_context=11, conv_padding='valid', mb_size=16,
                activation='relu', lirelu_alpha=.3, conv_stride=2,
                initialization='glorot_uniform', fast_text=False,
                batch_norm=True, dropout=False, stateful=False):

        logger.info("Building half phoneme model")
        assert self.model is None

        leaky_relu = False
        if activation == 'lirelu':
            activation = 'linear'
            leaky_relu = True

        if stateful:
            if mb_size is None:
                raise ValueError("Stateful GRU layer needs to know batch size")
            acoustic_input = Input(batch_shape=(mb_size, None, input_dim),
                                   name='acoustic_input')
        else:
            acoustic_input = Input(shape=(None, input_dim),
                                   name='acoustic_input')

        branch = 'phoneme'
        self.branch_vars[branch] = []
        conv_1dl = Conv1D(nodes, conv_context, name='conv_1d',
                          padding=conv_padding, strides=conv_stride,
                          kernel_initializer=initialization,
                          activation=activation)
        output = conv_1dl(acoustic_input)
        self.branch_vars[branch].extend(conv_1dl.trainable_weights)

        if batch_norm:
            bn_l = BatchNormalization(name='bn_conv_1d')
            output = bn_l(output)
            self.branch_vars[branch].extend(bn_l.trainable_weights)

        if leaky_relu:
            output = LeakyReLU(alpha=lirelu_alpha)(output)

        if dropout:
            output = Dropout(dropout)(output)

        for r in range(recur_layers):
            gru_l = GRU(nodes, activation=activation, stateful=stateful,
                        name='rnn_{}'.format(r + 1),
                        kernel_initializer=initialization,
                        return_sequences=True)
            output = gru_l(output)
            self.branch_vars[branch].extend(gru_l.trainable_weights)

            if batch_norm:
                bn_l = BatchNormalization(name='bn_rnn_{}'.format(r + 1),
                                          mode=2)
                output = bn_l(output)
                self.branch_vars[branch].extend(bn_l.trainable_weights)

            if leaky_relu:
                output = LeakyReLU(alpha=lirelu_alpha)(output)

            if r+1 == recur_layers // 2:
                phoneme_dense = Dense(
                    self.phono_dim, name='phoneme_dense',
                    activation=for_tf_or_th('softmax', 'linear'),
                    kernel_initializer=initialization)
                phoneme_branch = TimeDistributed(phoneme_dense, name=branch)
                phoneme_out = phoneme_branch(output)

                branch = 'text'
                if fast_text:
                    self.branch_vars[branch] = list(self.branch_vars['phoneme'])
                else:
                    self.branch_vars[branch] = []

        text_dense = Dense(self.vocab_dim, name='text_dense',
                           activation=for_tf_or_th('softmax', 'linear'),
                           kernel_initializer=initialization)
        text_branch = TimeDistributed(text_dense, name=branch)
        text_out = text_branch(output)

        self.branch_vars['phoneme'].extend(phoneme_branch.trainable_weights)
        self.branch_vars['text'].extend(text_branch.trainable_weights)

        self.model = Model(input=acoustic_input, output=[phoneme_out, text_out])
        self.branch_outputs = [phoneme_branch, text_branch]
        self.acoustic_input = self.model.inputs[0]
        return self.model


class TwoHornModelWrapper(ModelWrapper):

    def compile(self, input_dim=161, phoneme_recurs=2, nodes=1024,
                text_recurs=3, conv_context=11, conv_stride=2,
                conv_padding='valid', mb_size=16,
                initialization='glorot_uniform', stateful=False):
        assert self.model is None
        if stateful:
            if mb_size is None:
                raise ValueError("Stateful GRU layer needs to know batch size")
            acoustic_input = Input(batch_shape=(mb_size, None, input_dim),
                                   name='acoustic_input')
        else:
            acoustic_input = Input(shape=(None, input_dim),
                                   name='acoustic_input')

        branch = 'phoneme'
        self.branch_vars[branch] = []
        ph_conv1 = Conv1D(nodes, conv_context, name='ph_conv1',
                          padding=conv_padding, strides=conv_stride,
                          kernel_initializer=initialization, activation='relu')
        ph_output = ph_conv1(acoustic_input)
        self.branch_vars[branch].extend(ph_conv1.trainable_weights)

        bn_l = BatchNormalization(name='bn_ph_conv1')
        ph_output = bn_l(ph_output)
        self.branch_vars[branch].extend(bn_l.trainable_weights)

        for r in range(phoneme_recurs):
            gru_l = GRU(nodes, activation='relu', name='ph_rnn_{}'.format(r+1),
                        kernel_initializer=initialization,
                        stateful=stateful, return_sequences=True)
            ph_output = gru_l(ph_output)
            self.branch_vars[branch].extend(gru_l.trainable_weights)

            bn_l = BatchNormalization(name='bn_ph_rnn_{}'.format(r+1))
            ph_output = bn_l(ph_output)
            self.branch_vars[branch].extend(bn_l.trainable_weights)

        phoneme_dense = Dense(self.phono_dim, name='phoneme_dense',
                              activation=for_tf_or_th('softmax', 'linear'),
                              kernel_initializer=initialization)
        phoneme_branch = TimeDistributed(phoneme_dense, name=branch)
        phoneme_out = phoneme_branch(ph_output)

        branch = 'text'
        self.branch_vars[branch] = []

        tx_conv1 = Conv1D(nodes, conv_context, name='tx_conv1',
                          padding=conv_padding, strides=conv_stride,
                          kernel_initializer=initialization, activation='relu')
        tx_output = tx_conv1(acoustic_input)
        self.branch_vars[branch].extend(tx_conv1.trainable_weights)

        bn_l = BatchNormalization(name='bn_tx_conv1')
        tx_output = bn_l(tx_output)
        self.branch_vars[branch].extend(bn_l.trainable_weights)

        for r in range(text_recurs-1):
            gru_l = GRU(nodes, activation='relu', name='tx_rnn_{}'.format(r+1),
                        kernel_initializer=initialization,
                        stateful=stateful, return_sequences=True)
            tx_output = gru_l(tx_output)
            self.branch_vars[branch].extend(gru_l.trainable_weights)

            bn_l = BatchNormalization(name='bn_tx_rnn_{}'.format(r+1))
            tx_output = bn_l(tx_output)
            self.branch_vars[branch].extend(bn_l.trainable_weights)

        output = concatenate([ph_output, tx_output])

        mix_l = Dense(nodes, name='mix_dense', activation='linear',
                      kernel_initializer=initialization)
        output = mix_l(output)
        self.branch_vars[branch].extend(mix_l.trainable_weights)

        gru_l = GRU(nodes, activation='relu',
                    name='tx_rnn_{}'.format(text_recurs),
                    kernel_initializer=initialization, stateful=stateful,
                    return_sequences=True)
        output = gru_l(output)
        self.branch_vars[branch].extend(gru_l.trainable_weights)

        bn_l = BatchNormalization(name='bn_tx_rnn_{}'.format(text_recurs))
        output = bn_l(output)
        self.branch_vars[branch].extend(bn_l.trainable_weights)

        text_dense = Dense(self.vocab_dim, name='text_dense',
                           activation=for_tf_or_th('softmax', 'linear'),
                           kernel_initializer=initialization)
        text_branch = TimeDistributed(text_dense, name=branch)
        text_out = text_branch(output)

        self.branch_vars['phoneme'].extend(phoneme_branch.non_trainable_weights)
        self.branch_vars['text'].extend(text_branch.non_trainable_weights)

        self.model = Model(input=acoustic_input, output=[phoneme_out, text_out])
        self.branch_outputs = [phoneme_branch, text_branch]
        self.acoustic_input = self.model.inputs[0]
        return self.model


class ConvOverConvModelWrapper(ModelWrapper):
    """ Build a recurrent network (CTC) for speech with GRU units over
        multiple convolution layers"""

    def compile(self, conv_props, input_dim=161, recur_layers=3,
                nodes=1024, conv_border_mode='valid',
                initialization='glorot_uniform', batch_norm=True,
                stateful=False, mb_size=None):
        logger.info("Building gru model")
        assert self.model is None
        if stateful:
            if mb_size is None:
                raise ValueError("Stateful GRU layer needs to know batch size")
            acoustic_input = Input(batch_shape=(mb_size, None, input_dim),
                                   name='acoustic_input')
        else:
            acoustic_input = Input(shape=(None, input_dim),
                                   name='acoustic_input')

        # Setup the network
        output = acoustic_input
        for (c, (filters, size, stride)) in enumerate(conv_props):
            output = Conv1D(filters, size, name='conv_1d_{}'.format(c),
                            padding=conv_border_mode, strides=stride,
                            kernel_initializer=initialization,
                            activation='relu')(output)

            if batch_norm:
                output = BatchNormalization(name='bn_conv_1d', mode=2)(output)

        for r in range(recur_layers):
            output = GRU(nodes, activation='relu', name='rnn_{}'.format(r + 1),
                         kernel_initializer=initialization,
                         stateful=stateful, return_sequences=True)(output)
            if batch_norm:
                bn_layer = BatchNormalization(name='bn_rnn_{}'.format(r + 1),
                                              mode=2)
                output = bn_layer(output)

        output_branch = TimeDistributed(Dense(
            self.output_dim, name='dense', init=initialization,
            activation=for_tf_or_th('softmax', 'linear')
        ), name=self.outputs)
        network_output = output_branch(output)

        self.model = Model(input=acoustic_input, output=[network_output])
        self.branch_outputs = [output_branch]
        self.branch_vars[output_branch.name] = self.model.trainable_weights
        self.acoustic_input = self.model.inputs[0]
        return self.model

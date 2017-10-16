"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import numpy as np
import random
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

from concurrent.futures import ThreadPoolExecutor, wait

from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence
from arpabets import arpabet_to_int_sequence

RNG_SEED = 123
logger = logging.getLogger(__name__)


class DataGenerator(object):
    def __init__(self, step=10, window=20, max_freq=8000, desc_file=None,
                 use_arpabets=False, use_durations=False):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.use_arpabets = use_arpabets
        self.use_durations = use_durations

    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq)

    def load_metadata_from_desc_file(self, desc_file, partition='train',
                                     max_duration=10.0):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        logger.info('Reading description file: {} for partition: {}'
                    .format(desc_file, partition))
        audio_paths, durations, texts, arpabets = [], [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                    if self.use_arpabets:
                        arpabets.append(spec['arpabet'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or
                    # (KeyError,json.decoder.JSONDecodeError), depending on
                    # json module version
                    logger.warn('Error reading line #{}: {}'
                                .format(line_num, json_line))
                    logger.warn(str(e))

        if not self.use_arpabets:
            arpabets = [''] * len(audio_paths)

        if partition == 'train':
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
            self.train_arpabets = arpabets
        elif partition == 'validation':
            self.val_audio_paths = audio_paths
            self.val_durations = durations
            self.val_texts = texts
            self.val_arpabets = arpabets
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
            self.test_arpabets = arpabets
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")

    def load_train_data(self, desc_file, max_duration=10.0):
        self.load_metadata_from_desc_file(desc_file, 'train', max_duration)

    def load_test_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'test')

    def load_validation_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'validation')

    @staticmethod
    def sort_by_duration(durations, audio_paths, texts, arpabets):
        x = sorted(zip(durations, audio_paths, texts, arpabets))
        if K.backend() == 'theano':
            x.reverse()
        return zip(*x)

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def prepare_minibatch(self, audio_paths, texts, durations, arpabets):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts),\
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        # Calculate the features for each audio clip, as the log of the
        # Fourier Transform of the audio
        features = [self.featurize(a) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        y = []
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat
            label = text_to_int_sequence(texts[i])
            y.append(label)
            label_lengths.append(len(label))
        y = pad_sequences(y, maxlen=len(max(texts, key=len)), dtype='int32',
                          padding='post', truncating='post', value=-1)
        res = {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'y': y,  # list(int) Flattened labels (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths  # list(int) Length of each label
            # 'durations' [if use_durations] list(float) Duration of each sample
            # 'phonemes'[if use_arpabets] list(int) Flattened arpabet ints
        }
        if self.use_durations:
            res['durations'] = durations
        if self.use_arpabets:
            arpints, arpaint_lengths = [], []
            for i in range(mb_size):
                arpaint_seq = arpabet_to_int_sequence(arpabets[i])
                arpints.append(arpaint_seq)
                arpaint_lengths.append(len(arpaint_seq))
            maxlen = len(max(arpints, key=len))
            res['phonemes'] = pad_sequences(arpints, maxlen=maxlen,
                                            dtype='int32', padding='post',
                                            truncating='post', value=-1)
            res['phoneme_lengths'] = arpaint_lengths
        return res

    def iterate(self, audio_paths, texts, minibatch_size, durations=[],
                arpabets=[], max_iters=None, parallel=True):
        if max_iters is not None:
            k_iters = max_iters
        else:
            k_iters = int(np.ceil(len(audio_paths) / minibatch_size))
        logger.info("Iters: {}".format(k_iters))
        if parallel:
            pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
            future = pool.submit(self.prepare_minibatch,
                                 audio_paths[:minibatch_size],
                                 texts[:minibatch_size],
                                 durations[:minibatch_size],
                                 arpabets[:minibatch_size])
        else:
            minibatch = self.prepare_minibatch(audio_paths[:minibatch_size],
                                               texts[:minibatch_size],
                                               durations[:minibatch_size],
                                               arpabets[:minibatch_size])

        start = minibatch_size
        for i in range(k_iters - 1):
            if parallel:
                wait([future])
                minibatch = future.result()
                # While the current minibatch is being consumed, prepare the
                # next
                future = pool.submit(self.prepare_minibatch,
                                     audio_paths[start:start+minibatch_size],
                                     texts[start:start+minibatch_size],
                                     durations[start:start+minibatch_size],
                                     arpabets[start:start+minibatch_size])
            yield minibatch
            if not parallel:
                minibatch = self.prepare_minibatch(
                    audio_paths[start:start+minibatch_size],
                    texts[start:start+minibatch_size],
                    durations[start:start+minibatch_size],
                    arpabets[start:start+minibatch_size]
                )
            start += minibatch_size
        # Wait on the last minibatch
        if parallel:
            wait([future])
            minibatch = future.result()
            yield minibatch
        else:
            yield minibatch

    def iterate_train(self, minibatch_size=16, sort_by_duration=False,
                      shuffle=True):
        if sort_by_duration and shuffle:
            shuffle = False
            logger.warn("Both sort_by_duration and shuffle were set to True. "
                        "Setting shuffle to False")
        durations, audio_paths, texts, arpabets = (self.train_durations,
                                                   self.train_audio_paths,
                                                   self.train_texts,
                                                   self.train_arpabets)
        if shuffle:
            temp = zip(durations, audio_paths, texts, arpabets)
            self.rng.shuffle(temp)
            durations, audio_paths, texts, arpabets = zip(*temp)
        if sort_by_duration:
            logger.info('Sorting training samples by duration')
            if getattr(self, '_sorted_data', None) is None:
                self._sorted_data = DataGenerator.sort_by_duration(
                    durations, audio_paths, texts, arpabets)
            durations, audio_paths, texts, arpabets = self._sorted_data
        return self.iterate(audio_paths, texts, minibatch_size, durations,
                            arpabets)

    def iterate_test(self, minibatch_size=16):
        return self.iterate(self.test_audio_paths, self.test_texts,
                            minibatch_size, self.test_durations)

    def iterate_validation(self, minibatch_size=16):
        return self.iterate(self.val_audio_paths, self.val_texts,
                            minibatch_size, self.val_durations,
                            self.val_arpabets)

    def fit_train(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)

    def reload_norm(self, dataset):
        """ Set mean and std of features from previous calculations
        Params:
            dataset (str)
        """
        if dataset == '860-1000':
            self.feats_std = np.array([
                4.25136062, 3.8713157, 4.27721627, 4.79254968, 5.047769,
                5.00917253, 4.92034587, 4.95192179, 4.99958183, 4.98448796,
                4.93224872, 4.85590985, 4.78577772, 4.70706027, 4.62677301,
                4.54424163, 4.455477, 4.38643766, 4.32992825, 4.28711064,
                4.24306676, 4.24044366, 4.23590435, 4.21825687, 4.19820567,
                4.17238816, 4.12828632, 3.903265, 3.88530966, 4.10232629,
                4.15094822, 4.14674498, 4.13922566, 4.13210467, 4.12067026,
                4.10835004, 4.09651096, 4.08038286, 4.06577381, 4.04688416,
                4.01817645, 4.02679759, 4.02986556, 4.03453092, 4.04160862,
                4.04830856, 4.0602057, 4.0771961, 4.09297194, 4.11034371,
                4.11758663, 4.12095657, 4.11906109, 4.1155184, 4.10200893,
                4.08276379, 4.08628075, 4.08675451, 4.07840435, 4.06359915,
                4.04148782, 4.06030573, 4.06159643, 4.0473447, 4.03310411,
                4.02725498, 4.02498171, 4.02632823, 4.02484766, 4.02769822,
                4.02489051, 4.02088211, 4.02309526, 4.01872619, 4.01964194,
                4.02153504, 4.02851296, 4.02778547, 4.0279664, 4.02255787,
                4.00012165, 4.01658932, 3.93528177, 3.89534593, 4.017947,
                4.03439452, 4.03349856, 4.03254631, 4.03193693, 4.0297471,
                4.02667958, 4.02249605, 4.01419366, 4.01364902, 4.01290134,
                4.01051293, 4.0089972, 4.00612032, 4.00165361, 3.98616987,
                3.96209925, 3.98299328, 3.99713713, 3.99335162, 3.99078871,
                3.98656532, 3.98739388, 3.98590306, 3.99035434, 3.98769832,
                3.96287722, 3.97156738, 3.9831056, 3.97919869, 3.9740908,
                3.96782821, 3.96331332, 3.95866512, 3.94998412, 3.92881555,
                3.90712036, 3.92175492, 3.92782247, 3.92540498, 3.92125062,
                3.91851146, 3.91551745, 3.90756256, 3.90717957, 3.90416995,
                3.8988367, 3.89860032, 3.89073579, 3.8857745, 3.88896128,
                3.88531893, 3.87826128, 3.84122702, 3.80223124, 3.84022503,
                3.83359076, 3.85137669, 3.85647565, 3.85543909, 3.85545835,
                3.85710792, 3.85664199, 3.85589913, 3.85612751, 3.85606959,
                3.84913531, 3.84287049, 3.83940561, 3.84250948, 3.83669538,
                3.83200409, 3.83749335, 3.8358794, 3.83973577, 3.83577876,
                3.88315269])
            self.feats_mean = np.array([
                -19.9190417, -17.87074816, -17.18417253, -16.60615722,
                -16.47524177, -16.78722456, -17.22669022, -17.37899149,
                -17.43706583, -17.56693628, -17.78871635, -18.10035868,
                -18.49648794, -18.90612757, -19.25623952, -19.51816016,
                -19.7352671, -19.91201681, -20.07744978, -20.23590349,
                -20.36163737, -20.49420555, -20.59985973, -20.68975368,
                -20.76943646, -20.81912058, -20.83019722, -20.71162806,
                -20.69917742, -20.80050996, -20.80537402, -20.81130023,
                -20.8206865, -20.84339359, -20.87898781, -20.9239178,
                -20.97779635, -21.03484084, -21.11069463, -21.18509355,
                -21.23770916, -21.31738037, -21.37270682, -21.41930211,
                -21.4509242, -21.47946454, -21.50444787, -21.51825131,
                -21.51594888, -21.51341618, -21.50983657, -21.52451875,
                -21.55498166, -21.59505743, -21.63528843, -21.67396515,
                -21.73275922, -21.79585578, -21.84556578, -21.87028101,
                -21.87358228, -21.92915795, -21.98821111, -22.04652423,
                -22.10257075, -22.14423208, -22.17747545, -22.21208271,
                -22.24781483, -22.28908871, -22.33593842, -22.37691381,
                -22.42626383, -22.46079106, -22.48787287, -22.50766501,
                -22.53586539, -22.56786281, -22.59582998, -22.62581144,
                -22.64886813, -22.71750843, -22.7279599, -22.7583583,
                -22.87805837, -22.9381045, -22.98543052, -23.03572058,
                -23.09462637, -23.15124978, -23.20831305, -23.26313514,
                -23.3142818, -23.3671435, -23.41988972, -23.47336085,
                -23.52725449, -23.57844801, -23.63283689, -23.67411968,
                -23.69833849, -23.77693613, -23.83653636, -23.87643816,
                -23.91009798, -23.94392507, -23.98188049, -24.0217485,
                -24.05824361, -24.09093447, -24.1127368, -24.15196192,
                -24.18803859, -24.21636283, -24.24457024, -24.27694599,
                -24.3077978, -24.33267521, -24.35511158, -24.3535704,
                -24.33095828, -24.37981144, -24.41567347, -24.42952345,
                -24.42856408, -24.43132484, -24.44428014, -24.46147466,
                -24.48292062, -24.50661327, -24.53453117, -24.56691744,
                -24.58867348, -24.61431382, -24.6420865, -24.66867143,
                -24.70376135, -24.71120825, -24.70337552, -24.74880836,
                -24.74934962, -24.8100975, -24.85395296, -24.88296942,
                -24.90949419, -24.93485977, -24.96167982, -24.98794161,
                -25.01261378, -25.04233288, -25.06206523, -25.0873588,
                -25.11614341, -25.1463879, -25.16707626, -25.18893841,
                -25.22892228, -25.26932148, -25.31942099, -25.34896047,
                -26.51045921])
        else:
            raise ValueError

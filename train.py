"""
Train an end-to-end speech recognition model using CTC.
Use $python train.py --help for usage
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import sys
import os

from data_generator import DataGenerator
from utils import configure_logging

from model_wrp import HalfPhonemeModelWrapper
from trainer import Trainer

logger = logging.getLogger(__name__)


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


def main(train_desc_file, val_desc_file, epochs, save_dir, sortagrad,
         use_arpabets, start_weights=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Configure logging
    configure_logging(file_log_path=os.path.join(save_dir, 'train_log.txt'))
    logger.info(' '.join(sys.argv))

    # Prepare the data generator
    datagen = DataGenerator(use_arpabets=use_arpabets)
    # Load the JSON file that contains the dataset
    datagen.load_train_data(train_desc_file, max_duration=20)
    datagen.load_validation_data(val_desc_file)
    # Use a few samples from the dataset, to calculate the means and variance
    # of the features, so that we can center our inputs to the network
    # datagen.fit_train(100)
    datagen.reload_norm('860-1000')
    train_sample_half_phoneme(datagen, save_dir, epochs, sortagrad,
                              start_weights, mb_size=48)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'training labels and paths to the audio files.')
    parser.add_argument('val_desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'validation labels and paths to the audio files.')
    parser.add_argument('save_dir', type=str,
                        help='Directory to store the model. This will be '
                             'created if it doesn\'t already exist')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train the model')
    parser.add_argument('--sortagrad', default=False, nargs='?', const=True,
                        type=int, help='Sort utterances by duration for this '
                        'number of epochs. Will sort all epochs if no value '
                        'is given')
    parser.add_argument('--use-arpabets', default=False,
                        help='Read arpabets', action='store_true')
    parser.add_argument('--start-weights', type=str, default=None,
                        help='Load weights')
    args = parser.parse_args()

    main(args.train_desc_file, args.val_desc_file, args.epochs, args.save_dir,
         args.sortagrad, args.use_arpabets, args.start_weights)

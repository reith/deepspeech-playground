"""
Test a trained speech model over a dataset
"""

from __future__ import absolute_import, division, print_function
import argparse

import os
import sys
import json


def load_model_wrapper(model_config_file, weights_file):
    """Loads a pre-trained model wrapper"""
    with open(model_config_file) as fp:
        model_config = json.load(fp)
    try:
        os.environ['KERAS_BACKEND'] = model_config['backend']
        import model_wrp
        # pretrained_id = model_config['pre-trained-id']
        wrapper_config = model_config['model_wrapper']
        wrapper_class = getattr(sys.modules['model_wrp'],
                                wrapper_config['class_name'])
        model_wrapper = wrapper_class(**wrapper_config.get('init_args', {}))
        model = model_wrapper.compile(**wrapper_config.get('compile_args', {}))
        model.load_weights(weights_file)
    except (KeyError, ):
        print ("Model is not known")
        sys.exit(1)
    return model_wrapper


def main(test_desc_file, model_config_file, weights_file):
    # Load model
    model_wrapper = load_model_wrapper(model_config_file, weights_file)
    model, test_fn = model_wrapper.model, model_wrapper.compile_test_fn()

    # Prepare the data generator
    from data_generator import DataGenerator
    datagen = DataGenerator()
    # Load the JSON file that contains the dataset
    datagen.load_validation_data(test_desc_file)
    # Normalize input data by variance and mean of training input
    datagen.reload_norm('860-1000')

    from trainer import Trainer
    trainer = Trainer(model, None, test_fn)
    trainer.validate(datagen, 32, False, False, None)
    # Test the model
    print ("Test loss: {}".format(trainer.last_val_cost))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'test labels and paths to the audio files. ')
    parser.add_argument('model_config', type=str, help='Path to model config')
    parser.add_argument('weights_file', type=str,
                        help='Load weights from this file')
    args = parser.parse_args()
    main(args.test_desc_file, args.model_config, args.weights_file)

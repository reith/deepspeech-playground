r"""
Use this script to visualize the output of a trained speech-model.
Usage: python visualize.py /path/to/audio /path/to/training/json.json \
            /path/to/model
"""

from __future__ import absolute_import, division, print_function
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from test import load_model_wrapper


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def prompt_loop(prompt_line, locs):
    """ Reads user codes and evaluates them then returns new locals scope """

    while True:
        try:
            line = raw_input(prompt_line)
        except EOFError:
            break
        else:
            if line.strip() == '':
                break
        try:
            exec(line, globals(), locs)
        except Exception as exc:
            print(exc)
            continue

    return locs


def visualize(model, test_file, train_desc_file):
    """ Get the prediction using the model, and visualize softmax outputs
    Params:
        model (keras.models.Model): Trained speech model
        test_file (str): Path to an audio clip
        train_desc_file(str): Path to the training file used to train this
                              model
    """
    from model import compile_output_fn
    from data_generator import DataGenerator
    from utils import argmax_decode

    datagen = DataGenerator()
    datagen.load_train_data(train_desc_file)
    datagen.fit_train(100)

    print ("Compiling test function...")
    test_fn = compile_output_fn(model)

    inputs = [datagen.featurize(test_file)]

    prediction = np.squeeze(test_fn([inputs, True]))
    # preds, probs = beam_decode(prediction, 8)
    # u_preds, u_probs = beam_decode_u(prediction, 8)

    softmax_file = "softmax.npy".format(test_file)
    softmax_img_file = "softmax.png".format(test_file)
    print ("Prediction: {}"
           .format(argmax_decode(prediction)))
    print ("Saving network output to: {}".format(softmax_file))
    print ("As image: {}".format(softmax_img_file))
    np.save(softmax_file, prediction)
    sm = softmax(prediction.T)
    sm = np.vstack((sm[0], sm[2], sm[3:][::-1]))
    fig, ax = plt.subplots()
    ax.pcolor(sm, cmap=plt.cm.Greys_r)
    column_labels = [chr(i) for i in range(97, 97 + 26)] + ['space', 'blank']
    ax.set_yticks(np.arange(sm.shape[0]) + 0.5, minor=False)
    ax.set_yticklabels(column_labels[::-1], minor=False)
    plt.savefig(softmax_img_file)


def interactive_vis(model_dir, model_config, train_desc_file, weights_file=None):
    """ Get the prediction using the model, and visualize softmax outputs, able
    to predict multiple inputs.
    Params:
        model_dir (str): Trained speech model or None. If None given will ask
            code to make model.
        model_config (str): Path too pre-trained model configuration
        train_desc_file(str): Path to the training file used to train this
                              model
        weights_file(str): Path to stored weights file for model being made
    """

    if model_dir is None:
        assert weights_file is not None
        if model_config is None:
            from model_wrp import HalfPhonemeModelWrapper, GruModelWrapper
            print ("""Make and store new model into model, e.g.
                >>> model_wrp = HalfPhonemeModelWrapper()
                >>> model = model_wrp.compile(nodes=1000, recur_layers=5,
                                                conv_context=5)
                """)

            model = prompt_loop('[model=]> ', locals())['model']
            model.load_weights(weights_file)
        else:
            model_wrapper = load_model_wrapper(model_config, weights_file)
            test_fn = model_wrapper.compile_output_fn()
    else:
        from utils import load_model
        model = load_model(model_dir, weights_file)

    if model_config is None:
        print ("""Make and store test function to test_fn, e.g.
            >>> test_fn = model_wrp.compile_output_fn()
            """)
        test_fn = prompt_loop('[test_fn=]> ', locals())['test_fn']

    from utils import argmax_decode
    from data_generator import DataGenerator
    datagen = DataGenerator()

    if train_desc_file is not None:
        datagen.load_train_data(train_desc_file)
        datagen.fit_train(100)
    else:
        datagen.reload_norm('860-1000')

    while True:
        try:
            test_file = raw_input('Input file: ')
        except EOFError:
            comm_mode = True
            while comm_mode:
                try:
                    comm = raw_input("[w: load wieghts\t s: shell ] > ")
                    if comm.strip() == 'w':
                        w_path = raw_input("weights file path: ").strip()
                        model.load_weights(w_path)
                    if comm.strip() == 's':
                        prompt_loop('> ', locals())
                except EOFError:
                    comm_mode = False
                except Exception as exc:
                    print (exc)
            continue

        if test_file.strip() == '':
            break

        try:
            inputs = [datagen.normalize(datagen.featurize(test_file))]
        except Exception as exc:
            print (exc)
            continue

        prediction = np.squeeze(test_fn([inputs, False]))

        softmax_file = "softmax.npy".format(test_file)
        softmax_img_file = "softmax.png".format(test_file)
        print ("Prediction: {}".format(argmax_decode(prediction)))
        print ("Saving network output to: {}".format(softmax_file))
        print ("As image: {}".format(softmax_img_file))
        np.save(softmax_file, prediction)
        sm = softmax(prediction.T)
        sm = np.vstack((sm[0], sm[2], sm[3:][::-1]))
        fig, ax = plt.subplots()
        ax.pcolor(sm, cmap=plt.cm.Greys_r)
        column_labels = [chr(i) for i in range(97, 97+26)] + ['space', 'blank']
        ax.set_yticks(np.arange(sm.shape[0]) + 0.5, minor=False)
        ax.set_yticklabels(column_labels[::-1], minor=False)
        plt.savefig(softmax_img_file)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on input file(s).", epilog="""
        This script can give an interactive shell for evaluation on multiple
        input files. If you want plain prediction as originally came from
        Baidu's repo and model is trained without `model_wrapper` helpers,
        arguments --test-file, --train-desc-file, --load-dir and --weights-file
        are necessary.  Otherwise set --interactive and If model is shipped
        by this repo give model config by --model-config.
        """)
    parser.add_argument('--test-file', type=str, help='Path to an audio file')
    parser.add_argument('--train-desc-file', type=str,
                        help='Path to the training JSON-line file. This will '
                             'be used to extract feature means/variance')
    parser.add_argument('--load-dir', type=str,
                        help='Directory where a trained model is stored.')
    parser.add_argument('--model-config', type=str,
                        help='Path to pre-trained model configuration')
    parser.add_argument('--weights-file', type=str, default=None,
                        help='Path to a model weights file')
    parser.add_argument('--interactive', default=False, action='store_true',
                        help='Interactive interface, necessary for pre-trained'
                        ' models with this repo.')
    args = parser.parse_args()

    if args.interactive:
        assert args.test_file is None
        interactive_vis(args.load_dir, args.model_config, args.train_desc_file,
                        args.weights_file)
    else:
        from utils import load_model
        if args.load_dir is None or args.test_file is None:
            parser.print_usage()
            sys.exit(1)

        print ("Loading model")
        model = load_model(args.load_dir, args.weights_file)
        visualize(model, args.test_file, args.train_desc_file)


if __name__ == '__main__':
    main()

import os
import argparse
from image_complete import auto
import traceback

from sfp import Poller
from predict_common import prediction_to_file, load_model


SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []

    try:
        # TODO batches?
        imgs = []
        with open(fname, "rb") as fp:
            img_bytes = fp.read()
            imgs.append(img_bytes)
        preds = poller.params.engine.infer_raw(imgs)
        fname_out = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ".json")
        fname_out = prediction_to_file(preds[0], fname_out)
        result.append(fname_out)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(engine, input_dir, output_dir, tmp_dir,
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, verbose=False, quiet=False):
    """
    Method for performing predictions on images.

    :param engine: the CustomEngine to use
    :type engine: CustomEngine
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.progress = not quiet
    poller.verbose = verbose
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.engine = engine
    poller.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PaddleClas - Prediction", prog="paddleclas_predict_poll", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--model_path', help='Path to the trained model (.pdparams file), overrides config file', required=False, default=None)
    parser.add_argument('--class_id_map_file', help='Path to the file with the class index/label mapping, overrides config file', required=False, default=None)
    parser.add_argument('--device', help='The device to use', default="gpu")
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args()

    try:
        eng = load_model(parsed.config, model_path=parsed.model_path,
                         class_id_map_file=parsed.class_id_map_file,
                         device=parsed.device)

        # Performing the prediction and producing the predictions files
        predict_on_images(eng, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp,
                          continuous=parsed.continuous,
                          use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                          delete_input=parsed.delete_input, verbose=parsed.verbose, quiet=parsed.quiet)

    except Exception as e:
        print(traceback.format_exc())

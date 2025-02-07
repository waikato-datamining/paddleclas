from datetime import datetime
import numpy as np
import traceback
import cv2

from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import prediction_to_data, load_model
import paddle


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        array = np.frombuffer(msg_cont.message['data'], np.uint8)
        imgs = [array]
        preds = config.engine.infer_raw(imgs)
        out_data = prediction_to_data(preds[0])
        msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser('PaddleClas - Prediction (Redis)', prog="paddleclas_predict_redis", prefix="redis_")
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--model_path', help='Path to the trained model (.pdparams file), overrides config file', required=False, default=None)
    parser.add_argument('--class_id_map_file', help='Path to the file with the class index/label mapping, overrides config file', required=False, default=None)
    parser.add_argument('--device', help='The device to use', default="gpu")
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        eng = load_model(parsed.config, model_path=parsed.model_path,
                         class_id_map_file=parsed.class_id_map_file,
                         device=parsed.device)

        config = Container()
        config.engine = eng
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())

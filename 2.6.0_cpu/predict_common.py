import json
from typing import Tuple
from ppcls.engine.custom_engine import CustomEngine
from ppcls.utils import config


def load_model(config_path: str, model_path: str = None, class_id_map_file: str = None, device: str = "cpu") -> Tuple:
    """
    Loads the model.

    :param config_path: the path to the config file
    :type config_path: str
    :param model_path: the path to the trained model (.pdparams file), overrides config file
    :type model_path: str
    :param class_id_map_file: the path to the file with the class index/label mapping, overrides config file
    :type class_id_map_file: str
    :param device: the device to use, e.g., gpu or cpu
    :type device: str
    :return: the engine for performing inference
    :rtype: CustomEngine
    """
    cfg = config.get_config(config_path, show=False)
    if model_path is not None:
        cfg["Global"]["pretrained_model"] = model_path
    if class_id_map_file is not None:
        if "Infer" not in cfg:
            cfg["Infer"] = dict()
        if "PostProcess" not in cfg["Infer"]:
            cfg["Infer"]["PostProcess"] = dict()
        cfg["Infer"]["PostProcess"]["class_id_map_file"] = class_id_map_file
    cfg["Global"]["device"] = device
    engine = CustomEngine(cfg, mode="infer")
    return engine


def prediction_to_file(prediction, path: str) -> str:
    """
    Saves the predictions to disk as JSON file.

    :param prediction: the paddleclas prediction object
    :param path: the path to save the image to
    :type path: str
    :return: the filename the predictions were saved under
    :rtype: str
    """
    content = prediction_to_data(prediction)
    with open(path, "w") as fp:
        fp.write(content)
        fp.write("\n")
    return path


def prediction_to_data(prediction) -> str:
    """
    Turns the mask prediction into bytes using the specified image format.

    :param prediction: the paddleclas prediction object
    :return: the generated JSON with the class probabilities
    :rtype: str
    """
    result = {}
    for i in range(len(prediction["label_names"])):
        result[prediction["label_names"][i]] = prediction["scores"][i]
    return json.dumps(result)

import argparse
import os.path
import traceback
import yaml
from typing import List, Optional, Any


def check_file(file_type: str, path: Optional[str]):
    """
    Checks whether file exists and does not point to a directory.
    Raises an exception otherwise.

    :param file_type: the string describing the file, used in the exceptions
    :type file_type: str
    :param path: the path to check, checks ignored if None
    :type path: str
    """
    if path is not None:
        if not os.path.exists(path):
            raise IOError("%s not found: %s" % (file_type, path))
        if os.path.isdir(path):
            raise IOError("%s points to a directory: %s" % (file_type, path))


def is_bool(s) -> bool:
    """
    Checks whether the string is a boolean value.

    :param s: the object/string to check
    :return: True if a boolean
    :rtype: bool
    """
    return (str(s).lower() == "true") or (str(s).lower() == "false")


def parse_bool(s: str) -> bool:
    """
    Returns True if the lower case of the string is "true".

    :param s: the string to evaluate
    :type s: str
    :return: True if "true"
    :rtype: bool
    """
    return str(s).lower() == "true"


def is_int(s) -> bool:
    """
    Checks whether the object/string is an int value.

    :param s: the objct/string to check
    :return: True if an int
    :rtype: bool
    """
    try:
        int(str(s))
        return True
    except:
        return False


def is_float(s) -> bool:
    """
    Checks whether the object/string is a float value.

    :param s: the object/string to check
    :return: True if a float
    :rtype: bool
    """
    try:
        float(str(s))
        return True
    except:
        return False


def set_value(config: dict, path: List[str], value: Any):
    """
    Sets the value in the YAML config according to its path.

    :param config: the config dictionary to update
    :type config: dict
    :param path: the list of path elements to use for navigating the hierarchical dictionary
    :type path: list
    :param value: the value to use
    """
    try:
        current = config
        found = False
        for i in range(len(path)):
            if path[i] in current:
                if i < len(path) - 1:
                    current = current[path[i]]
                else:
                    found = True
                    if isinstance(current[path[i]], bool):
                        current[path[i]] = parse_bool(value)
                    elif isinstance(current[path[i]], int):
                        current[path[i]] = int(value)
                    elif isinstance(current[path[i]], float):
                        current[path[i]] = float(value)
                    elif isinstance(current[path[i]], list):
                        values = value.split(",")
                        # can we infer type?
                        if len(current[path[i]]) > 0:
                            if isinstance(current[path[i]][0], bool):
                                current[path[i]] = [parse_bool(x) for x in values]
                            elif isinstance(current[path[i]][0], int):
                                current[path[i]] = [int(x) for x in values]
                            elif isinstance(current[path[i]][0], float):
                                current[path[i]] = [float(x) for x in values]
                            else:
                                current[path[i]] = values
                    else:
                        current[path[i]] = value
            elif path[i].startswith("[") and path[i].endswith("]") and isinstance(current, list):
                index = int(path[i][1:len(path[i])-1])
                if index < len(current):
                    current = current[index]
            else:
                # not present, we'll just add it
                if i == len(path) - 1:
                    print("Adding option: %s" % (str(path)))
                    if is_bool(value):
                        current[path[i]] = parse_bool(value)
                    elif is_int(value):
                        current[path[i]] = int(value)
                    elif is_float(value):
                        current[path[i]] = float(value)
                    else:
                        current[path[i]] = value
                    found = True
                break
        if not found:
            print("Failed to locate path in config: %s" % str(path))
    except:
        print("Failed to set value '%s' for path: %s" % (str(value), str(path)))
        traceback.print_stack()


def remove_value(config: dict, path: List[str]):
    """
    Removes the value from the YAML config according to its path.

    :param config: the config dictionary to update
    :type config: dict
    :param path: the list of path elements to use for navigating the hierarchical dictionary
    :type path: list
    """
    current = config
    removed = False
    for i in range(len(path)):
        if path[i] in current:
            if i < len(path) - 1:
                current = current[path[i]]
            else:
                del current[path[i]]
                removed = True
        elif path[i].startswith("[") and path[i].endswith("]") and isinstance(current, list):
            index = int(path[i][1:len(path[i])-1])
            if index < len(current):
                current = current[index]
        else:
            break
    if not removed:
        print("Failed to locate path in config, cannot remove: %s" % str(path))


def export(input_file: str, output_file: str, train_annotations: str = None, val_annotations: str = None,
           num_classes: int = None, num_epochs: int = None, eval_interval: int = None, save_interval: int = None,
           label_map: str = None, output_dir: str = None, additional: List[str] = None, remove: List[str] = None,
           no_force_chwimage: bool = False):
    """
    Exports the config file while updating specified parameters.

    :param input_file: the template YAML config file to load and modify
    :type input_file: str
    :param output_file: the YAML file to store the updated config data in
    :type output_file: str
    :param train_annotations: the text file with the training annotations/images relation, ignored if None
    :type train_annotations: str
    :param val_annotations: the text file with the validation annotations/images relation, ignored if None
    :type val_annotations: str
    :param label_map: the text file with the label index/text mapping (format: 'N STR'; one per line, index N starts at 0), ignored if None
    :type label_map: str
    :param num_classes: the number of classes in the dataset, ignored if None
    :type num_classes: int
    :param num_epochs: the number of epochs to train, ignored if None
    :type num_epochs: int
    :param eval_interval: the interval to perform evaluation, ignored if None
    :type eval_interval: int
    :param save_interval: the interval to save the model, ignored if None
    :type save_interval: int
    :param output_dir: the directory where to store all the output in, ignored if None
    :type output_dir: str
    :param additional: the list of additional parameters to set, format: PATH:VALUE, with PATH being the dot-notation path through the YAML parameter hierarchy in the file; if VALUE is to update a list, then the elements must be separated by comma
    :type additional: list
    :param remove: the list of parameters to remove, format: PATH, with PATH being the dot-notation path through the YAML parameter hierarchy in the file
    :type remove: list
    :param no_force_chwimage: disables enforcing the 'ToCHWImage' transform for inference
    :type no_force_chwimage: bool
    """
    # some sanity checks
    check_file("Config file", input_file)
    check_file("Training annotations", train_annotations)
    check_file("Validation annotations", val_annotations)
    if (num_classes is not None) and (num_classes < 1):
        num_classes = None

    # load template
    print("Loading config from: %s" % input_file)
    with open(input_file, 'r') as fp:
        config = yaml.safe_load(fp)

    if train_annotations is not None:
        set_value(config, ["DataLoader", "Train", "dataset", "name"], "ImageNetDataset")
        set_value(config, ["DataLoader", "Train", "dataset", "cls_label_path"], train_annotations)
        set_value(config, ["DataLoader", "Train", "dataset", "image_root"], os.path.dirname(train_annotations))

    if val_annotations is not None:
        set_value(config, ["DataLoader", "Eval", "dataset", "name"], "ImageNetDataset")
        set_value(config, ["DataLoader", "Eval", "dataset", "cls_label_path"], val_annotations)
        set_value(config, ["DataLoader", "Eval", "dataset", "image_root"], os.path.dirname(val_annotations))

    if label_map is not None:
        set_value(config, ["Infer", "PostProcess", "class_id_map_file"], label_map)

    # ensure that we have ToCHWImage transform at inference time?
    if not no_force_chwimage:
        if "Infer" in config:
            if "transforms" in config["Infer"]:
                present = False
                for i in config["Infer"]["transforms"]:
                    if "ToCHWImage" in i:
                        present = True
                        break
                if not present:
                    config["Infer"]["transforms"].append({"ToCHWImage": {}})

    if num_classes is not None:
        set_value(config, ["Arch", "class_num"], num_classes)

    if num_epochs is not None:
        set_value(config, ["Global", "epochs"], num_epochs)

    if eval_interval is not None:
        set_value(config, ["Global", "eval_interval"], eval_interval)

    if save_interval is not None:
        set_value(config, ["Global", "save_interval"], save_interval)

    if output_dir is not None:
        set_value(config, ["Global", "output_dir"], output_dir)
        set_value(config, ["Global", "save_inference_dir"], os.path.join(output_dir, "inference"))

    if additional is not None:
        for add in additional:
            if ":" in add:
                path_str, value = add.split(":")
                path = path_str.split(".")
                set_value(config, path, value)
            else:
                print("Invalid format for additional parameter, expected PATH:VALUE but found: %s" % add)

    if remove is not None:
        for rem in remove:
            path = rem.split(".")
            remove_value(config, path)

    print("Saving config to: %s" % output_file)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as fp:
        yaml.dump(config, fp)


def main(args=None):
    """
    Performs the bash.bashrc generation.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description='Exports a PaddleClas config file and updates specific fields with user-supplied values.',
        prog="paddleclas_export_config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", metavar="FILE", required=True, help="The PaddleClass YAML config file template to export.")
    parser.add_argument("-o", "--output", metavar="FILE", required=True, help="The YAML file to store the exported config file in.")
    parser.add_argument("-O", "--output_dir", metavar="DIR", required=False, help="The directory to store all the training output in.")
    parser.add_argument("-t", "--train_annotations", metavar="FILE", required=False, help="The text file with the labels for the training data (images are expected to be located below that directory).")
    parser.add_argument("-v", "--val_annotations", metavar="FILE", required=False, help="The text file with the labels for the validation data (images are expected to be located below that directory).")
    parser.add_argument("-l", "--label_map", metavar="FILE", required=False, help="The text file with the label index/text mapping (format: 'N STR'; one per line, index N starts at 0).")
    parser.add_argument("-c", "--num_classes", metavar="NUM", required=False, type=int, help="The number of classes in the dataset.")
    parser.add_argument("-e", "--num_epochs", metavar="NUM", required=False, type=int, help="The number of epochs to train.")
    parser.add_argument("--eval_interval", metavar="NUM", required=False, type=int, help="The number of epochs after which to perform an evaluation.")
    parser.add_argument("--save_interval", metavar="NUM", required=False, type=int, help="The number of epochs after which to save the current model.")
    parser.add_argument("-a", "--additional", metavar="PATH:VALUE", required=False, help="Additional parameters to override; format: PATH:VALUE, with PATH representing the dot-notation path through the parameter hierarchy in the YAML file, if VALUE is to update a list, then the elements must be separated by comma.", nargs="*")
    parser.add_argument("-r", "--remove", metavar="PATH", required=False, help="Parameters to remove; format: PATH, with PATH representing the dot-notation path through the parameter hierarchy in the YAML file", nargs="*")
    parser.add_argument("--no_force_chwimage", action="store_true", help="Does not enforce the 'ToCHWImage' transform for inference.")
    parsed = parser.parse_args(args=args)
    export(parsed.input, parsed.output,
           train_annotations=parsed.train_annotations, val_annotations=parsed.val_annotations,
           label_map=parsed.label_map, num_classes=parsed.num_classes, num_epochs=parsed.num_epochs,
           output_dir=parsed.output_dir, eval_interval=parsed.eval_interval, save_interval=parsed.save_interval,
           additional=parsed.additional, remove=parsed.remove, no_force_chwimage=parsed.no_force_chwimage)


def sys_main():
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    :rtype: int
    """

    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())

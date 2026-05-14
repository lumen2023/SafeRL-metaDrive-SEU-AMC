import os.path as osp
import torch
import yaml

def load_config_and_model(
    path: str,
    best: bool = False,
    config_path: str = None,
    model_path: str = None,
):
    """
    Load the configuration and trained model from a specified directory.

    :param path: the directory path where the configuration and trained model are stored.
    :param best: whether to load the best-performing model or the most recent one.
        Defaults to False.
    :param config_path: optional explicit path to the config file.
    :param model_path: optional explicit path to the checkpoint file.

    :return: a tuple containing the configuration dictionary and the trained model.
    :raises ValueError: if the specified directory does not exist.
    """
    if osp.exists(path):
        config_file = config_path or osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        if model_path is None:
            model_file = "model_best.pt" if best else "model.pt"
            model_path = osp.join(path, "checkpoint", model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path, weights_only=False)
        return config, model
    else:
        raise ValueError(f"{path} doesn't exist!")

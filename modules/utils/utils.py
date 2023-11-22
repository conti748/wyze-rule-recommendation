import yaml


def get_config(file_path: str) -> dict:
    """"
    Read the config file
    """
    with open(file_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

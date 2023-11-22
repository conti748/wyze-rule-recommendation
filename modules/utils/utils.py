import yaml


def get_config(file_path: str) -> dict:
    """"
    Read the config file
    """
    with open(file_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def find_rank(res, gt):
    try:
        index = res.index(gt)
        return index + 1
    except ValueError:
        return None

import yaml


def parse_config(config_file):
    """
    Helper function to parse the Yaml config file.

    Args:
        config_file (str): yml configuration file

    Returns:
        dict: parsed dictionary object
    """
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

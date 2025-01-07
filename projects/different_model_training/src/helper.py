import yaml

def get_global_config():
    # get global config information
    config = None
    with open("config/config.yml") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    return config

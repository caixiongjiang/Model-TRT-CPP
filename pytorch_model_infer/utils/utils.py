import yaml


def config_read():
    config_path = "./config/config.yaml"
    fo = open(config_path, 'r', encoding='utf-8')
    res = yaml.load(fo, Loader=yaml.FullLoader) 

    return res


def return_config():
    res = config_read()
    return res["LOG"], res["MODEL"], res["SERVER"]

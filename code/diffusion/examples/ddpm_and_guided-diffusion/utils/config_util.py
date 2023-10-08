import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            flag = 1
            for key_2 in value.keys():
                if not isinstance(key_2, str):
                    flag = 0
            if flag==1:
                new_value = dict2namespace(value)
            else:
                new_value = value
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
import torch
from collections import OrderedDict
import numpy as np
import os
import logging.config
import random
import yaml
import re
import logging


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_yaml_(args):
    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value


# dynamic adjust values of args
def load_temp_(args):
    def list2dict(l):
        d = {}
        for arg in l:
            k, v = arg.split('=')
            d[k] = v
        return d

    if args.temporary is not None:
        d = list2dict(args.temporary)
        for k, v in d.items():
            type_v = type(vars(args)[k])
            if type_v == int:
                vars(args)[k] = int(v)
            elif type_v == float:
                vars(args)[k] = float(v)
            elif type_v == bool:
                vars(args)[k] = bool(v)
            elif type_v == str:
                vars(args)[k] = str(v)


def config_logging(log_file='imgnet_log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode
                        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_state_from_ddp(model, state_dict):
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k[:7]:
            # remove `module.`
            name = k[7:]
            # print('loading {}'.format(name))
        elif 'backbone' in k:
            name = k.replace('backbone.', '')
        else:
            name = k
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.to('cuda')
    return model


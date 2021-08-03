# Minimal code for operation and understand

import torch
import os
import imp
import time
import datetime
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.utils.general_utils import dummy_context, AttrDict, get_clipped_optimizer, \
                                                        AverageMeter, ParamDict
from spirl.components.params import get_args


use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('configs/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def _default_hparams():
    default_dict = ParamDict({
        'model': None,
        'model_test': None,
        'logger': None,
        'logger_test': None,
        'evaluator': None,
        'data_dir': None,  # directory where dataset is in
        'batch_size': 128,
        'exp_path': None,  # Path to the folder with experiments
        'num_epochs': 200,
        'epoch_cycles_train': 1,
        'optimizer': 'radam',  # supported: 'adam', 'radam', 'rmsprop', 'sgd'
        'lr': 1e-3,
        'gradient_clip': None,
        'momentum': 0,  # momentum in RMSProp / SGD optimizer
        'adam_beta': 0.9,  # beta1 param in Adam
        'top_of_n_eval': 1,  # number of samples used at eval time
        'top_comp_metric': None,  # metric that is used for comparison at eval time (e.g. 'mse')
    })
    return default_dict


def get_exp_dir():
    return os.environ['EXP_DIR']


def get_config(args):
    conf = AttrDict()

    # paths
    conf.exp_dir = get_exp_dir()
    conf.conf_path = get_config_path(args.path)

    # general and model configs
    print('loading from the config file {}'.format(conf.conf_path))
    conf_module = imp.load_source('conf', conf.conf_path)
    conf.general = conf_module.configuration
    conf.model = conf_module.model_config

    # data config
    try:
        data_conf = conf_module.data_config
    except AttributeError:
        data_conf_file = imp.load_source('dataset_spec', os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
        data_conf = AttrDict()
        data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
        data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
    conf.data = data_conf

    # model loading config
    conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

    return conf


def get_dataset():
    pass
    # loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)


def get_model():
    print("use_cude: {}, device: {}".format(use_cuda, device))
    args = get_args()
    args.path = '/home/twkim/cloudrobot/spirl/spirl/configs/skill_prior_learning/kitchen/hierarchical'
    conf = get_config(args=args)
    _hp = _default_hparams()
    _hp.overwrite(conf.general)  # override defaults with config file
    _hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
    log_dir = log_dir = os.path.join(_hp.exp_path, 'events')
    print('using log dir: ', log_dir)

    conf.model['batch_size'] = _hp.batch_size
    conf.model.update(conf.data.dataset_spec)
    conf.model['device'] = conf.data['device'] = device.type

    train_params = AttrDict(logger_class=_hp.logger,
                            model_class=_hp.model,
                            n_repeat=_hp.epoch_cycles_train,
                            dataset_size=-1)
    print(train_params.model_class)
    logger = None
    start = time.time()
    model = train_params.model_class(conf.model, logger)
    print("elapsed time for model: {}".format(time.time() - start))
    print("model info: ", model)


def do_learning():
    pass


if __name__ == '__main__':
    get_model()
    # get_dataset()

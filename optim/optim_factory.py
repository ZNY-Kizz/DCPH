from torch import optim as optim
from optim import Nadam, RMSpropTF

def add_weight_decay(model_named_params, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model_named_params:
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(cfg, model_params, model_named_params, filter_bias_and_bn=True):
    if cfg.weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model_named_params, cfg.weight_decay)
        weight_decay = 0.
    else:
        parameters = model_params

    if cfg.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=cfg.lr,
            momentum=cfg.momentum, weight_decay=weight_decay, nesterov=True)
    elif cfg.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            parameters, lr=cfg.lr, weight_decay=weight_decay, eps=cfg.opt_eps)
    elif cfg.optimizer.lower() == 'nadam':
        optimizer = Nadam(
            parameters, lr=cfg.lr, weight_decay=weight_decay, eps=cfg.opt_eps)
    elif cfg.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=cfg.lr, weight_decay=weight_decay, eps=cfg.opt_eps)
    elif cfg.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=cfg.lr, alpha=0.9, eps=cfg.opt_eps,
            momentum=cfg.momentum, weight_decay=weight_decay)
    elif cfg.optimizer.lower() == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, lr=cfg.lr, alpha=0.9, eps=cfg.opt_eps,
            momentum=cfg.momentum, weight_decay=weight_decay)
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    return optimizer

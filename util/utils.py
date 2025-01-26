import logging

import wandb
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def log_metric(names: list, values, round: int, log: bool = False):
    labels = [name + ' : ' + str(value) for name, value in zip(names, values)]
    if log:
        print(', '.join(labels), ", round: ", round)
    wandb.log({name: value for name, value in zip(names, values)}, step=round)

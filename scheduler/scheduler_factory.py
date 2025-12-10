from scheduler.cosine_lr import CosineLRScheduler
from scheduler.plateau_lr import PlateauLRScheduler
from scheduler.tanh_lr import TanhLRScheduler
from scheduler.step_lr import StepLRScheduler
from torch.optim.lr_scheduler import *


def create_scheduler(cfg, num_epochs, optimizer):
    if cfg.scheduler == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=1e-5,
            decay_rate=cfg.decay_rate,
            warmup_lr_init=cfg.warmup_lr,
            warmup_t=cfg.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + 10
        # lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
    elif cfg.scheduler== 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=1e-7,
            warmup_lr_init=cfg.warmup_lr,
            warmup_t=cfg.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + 10
    elif cfg.scheduler == 'exp':
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=cfg.decay_rate)
    elif cfg.scheduler == 'multistep':
        assert type(cfg.decay_epochs) == list
        print(">>>> multistep decay: {}, decay_rate: {}".format(cfg.decay_epochs, cfg.decay_rate))
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=cfg.decay_epochs, gamma=cfg.decay_rate)
    else:
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=cfg.decay_epochs[0],
            decay_rate=cfg.decay_rate,
            warmup_lr_init=cfg.warmup_lr,
            warmup_t=cfg.warmup_epochs,
        )
    return lr_scheduler, num_epochs

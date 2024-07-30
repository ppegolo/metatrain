from torch.optim.lr_scheduler import LRScheduler


class WarmupLRSchedulerWrapper(LRScheduler):
    """
    Wraps the arbitrary PyTorch LR scheduler with a warmup scheduler.
    Starts from a very small learning rate (default: 1e-8) and linearly
    increases it to the value defined in the optimizer.

    After the `warmup_steps` epochs, the wrapped scheduler is used.

    :param lr_scheduler: The PyTorch LR scheduler to wrap.
    :param initial_lr: The initial learning rate for the warmup.
    :param warmup_steps: The number of steps for the warmup.

    """

    def __init__(
        self,
        lr_scheduler: LRScheduler,
        initial_lr: float = 1e-8,
        warmup_steps: int = 50,
        last_epoch=-1,
    ):
        self.finished_warmup = False
        self.warmup_steps = warmup_steps
        self.lr_scheduler = lr_scheduler
        self.initial_lrs = [initial_lr] * len(lr_scheduler.optimizer.param_groups)
        super(WarmupLRSchedulerWrapper, self).__init__(
            lr_scheduler.optimizer, last_epoch
        )
        self.last_epoch = last_epoch

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                initial_lr
                + (base_lr - initial_lr) * (self.last_epoch + 1) / self.warmup_steps
                for initial_lr, base_lr in zip(self.initial_lrs, self.base_lrs)
            ]
        else:
            self.finished_warmup = True
            return [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.get_lr()
        else:
            return self.lr_scheduler.get_last_lr()

    def step(self, metrics=None, epoch=None):
        if not self.finished_warmup:
            super(WarmupLRSchedulerWrapper, self).step(epoch)
        else:
            if epoch is None:
                self.lr_scheduler.step(metrics)
            else:
                self.lr_scheduler.step(metrics, epoch - self.warmup_steps)

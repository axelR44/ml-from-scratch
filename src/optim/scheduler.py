import numpy as np

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.epoch = 0

    def step(self):
        self.epoch += 1
        self.update_lr()

    def update_lr(self):
        raise NotImplementedError
    
class StepLR(LRScheduler):
     def __init__(self, optimizer, step_size=100, gamma=0.5):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

     def update_lr(self):
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma=0.99):
        super().__init__(optimizer)
        self.gamma = gamma

    def update_lr(self):
        self.optimizer.lr *= self.gamma


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max=500, eta_min=1e-5):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = optimizer.lr

    def update_lr(self):
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.epoch / self.T_max)
        ) / 2

        self.optimizer.lr = lr

class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, total_epochs, eta_min=1e-6):
        super().__init__(optimizer)
        self.total_epochs = total_epochs
        self.warmup_epochs = total_epochs/10
        self.eta_min = eta_min

        self.initial_lr = optimizer.lr
        self.epoch = 0

    def update_lr(self):
        if self.epoch < self.warmup_epochs:
            lr = self.initial_lr * (self.epoch / self.warmup_epochs)
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

            lr = self.eta_min + (self.initial_lr - self.eta_min) * (
                1 + np.cos(np.pi * progress)
            ) / 2

        self.optimizer.lr = lr
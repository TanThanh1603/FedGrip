from torch.optim import Optimizer


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    def step(self, beta=0):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                rate = beta if beta else group["lr"]
                parameter.data.add_(parameter.grad.data, alpha=-rate)

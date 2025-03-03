import torch


class NSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, dim):
        defaults = dict(lr=lr)
        super(NSGD, self).__init__(params, defaults)
        self.dim = dim

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad / (torch.norm(p.grad, dim=self.dim, keepdim=True) + 1e-5)
                p.add_(d_p, alpha=-group["lr"])
        return loss

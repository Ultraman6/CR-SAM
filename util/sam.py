import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self.grad_norm()
        # 用来存储本轮下降的原始梯度（未归一化且未放大rho倍）
        grads = []
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                # 收集原始梯度（未放大rho）
                grads.append(p.grad.data.clone().flatten())
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()
        # 返回收集的原始梯度
        return torch.cat(grads)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # 用来收集本轮更新时需要下降的梯度
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # 收集本轮下降的梯度（即梯度的副本）
                grads.append(p.grad.data.clone().flatten())
                # 更新梯度：返回到原始的 "w" 从 "w + e(w)"
                p.sub_(self.state[p]["e_w"])
        # 执行基础优化器的更新步骤（真正的"sharpness-aware"更新）
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
        # 返回收集的需要更新的梯度
        return torch.cat(grads)

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

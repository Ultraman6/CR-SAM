import torch


class SRAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SRAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, loss, zero_grad=False):
        """
        Perform the first step of SAM with log gradient perturbation.
        :param loss: The current loss to be used for log-gradient perturbation
        :param zero_grad: Whether to zero the gradients after this step
        """
        # Take the logarithm of the loss
        log_loss = torch.log(loss + 1e-12)  # Avoid log(0) by adding epsilon
        # Compute gradients of log(loss)
        log_loss.backward()
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # Avoid division by 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale  # Perturbation based on log-gradient
                p.add_(e_w)  # Move to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        # 基础优化器更新
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

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

import math
import torch
from torch.optim.optimizer import Optimizer

# TODO: see if we couldn't delegate to pytorch implementation instead of pasting it here
# TODO: fix https://github.com/pytorch/pytorch/pull/22628
# TODO: add schedule, wd normalization and warm restarts as in https://arxiv.org/pdf/1711.05101.pdf


class DenseSparseAdamW(Optimizer):
    """AdamW for dense parameters, SparseAdam for sparse parameters.

    The weight decay for sparse parameters is only applied to the updated
    weights.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                if grad.is_sparse:
                    # SparseAdam
                    grad = (
                        grad.coalesce()
                    )  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(
                        1 - beta1
                    )
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = (
                        grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    )
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group["eps"])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = (
                        group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                    )

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))

                else:
                    # AdamW
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    # according to the paper, this penalty should come after the bias correction
                    # if group['weight_decay'] != 0:
                    #     grad = grad.add(group['weight_decay'], p.data)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = (
                        group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                    )

                    p.data.addcdiv_(-step_size, exp_avg, denom)

                    if group["weight_decay"] != 0:
                        # Perform stepweight decay
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])

        return loss

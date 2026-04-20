from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # inputs: (..., vocab), targets: (...,)
    # Numerically stable version: log_softmax then gather
    x_max = inputs.amax(dim=-1, keepdim=True)
    x = inputs - x_max
    log_sum_exp = torch.log(torch.exp(x).sum(dim=-1, keepdim=True))
    log_probs = x - log_sum_exp
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -gathered.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                lr_t = lr * math.sqrt(bias_c2) / bias_c1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-lr_t)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    n = len(dataset)
    max_start = n - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])
    x_t = torch.from_numpy(x).long().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    return x_t, y_t


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src, weights_only=False)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

"""Triton implementation of the off by one softmax: https://www.evanmiller.org/attention-is-off-by-one.html.

This implementation is a derived form of the Liger-Kernel Softmax.
Credit: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/softmax.py
"""

import functools
from typing import Tuple

import torch
import triton
import triton.language as tl


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def is_hip() -> bool:
    return torch.version.hip is not None


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        msg = f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        raise RuntimeError(
            msg,
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _softmax_single_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(
        X_ptr + row_id * X_row_stride + offs,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".ca",
    )
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / (1 + d)
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask, cache_modifier=".cs")


@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    m = tl.float32(-float("inf"))
    d = tl.float32(0.0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(
            X_ptr + row_id * X_row_stride + idx,
            mask=mask,
            other=-float("inf"),
            cache_modifier=".ca",
        )
        blk_max = tl.max(xblk, axis=0)
        new_m = tl.max(m, blk_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
        m = new_m

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(
            X_ptr + row_id * X_row_stride + idx,
            mask=mask,
            other=-float("inf"),
            cache_modifier=".ca",
        )
        yblk = tl.exp(xblk - m) / (1 + d)
        tl.store(
            Y_ptr + row_id * Y_row_stride + idx,
            yblk,
            mask=mask,
            cache_modifier=".cs",
        )


@triton.jit
def _softmax_single_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    dy = tl.load(dy_ptr + row_id * dy_stride + offs, mask=mask, other=0.0)
    y = tl.load(
        y_ptr + row_id * y_stride + offs,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    )
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    tl.store(dx_ptr + row_id * dx_stride + offs, dx, mask=mask, cache_modifier=".wb")


@triton.jit
def _softmax_multi_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.float32(0.0)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(
            y_ptr + row_id * y_stride + idx,
            mask=mask,
            other=0.0,
            cache_modifier=".ca",
        )
        acc += tl.sum(dy_blk * y_blk, axis=0)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(
            y_ptr + row_id * y_stride + idx,
            mask=mask,
            other=0.0,
            cache_modifier=".ca",
        )
        dx_blk = y_blk * (dy_blk - acc)
        tl.store(
            dx_ptr + row_id * dx_stride + idx,
            dx_blk,
            mask=mask,
            cache_modifier=".wb",
        )


def _softmax_forward(x: torch.Tensor) -> tuple[torch.Tensor, int, int, bool]:
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y2d = torch.empty_like(x2d)

    if n_cols <= BLOCK_SIZE:
        _softmax_single_block_forward_kernel[(n_rows,)](
            y2d,
            y2d.stride(0),
            x2d,
            x2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        multi_block_launch = False
    else:
        _softmax_multi_block_forward_kernel[(n_rows,)](
            y2d,
            y2d.stride(0),
            x2d,
            x2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        multi_block_launch = True

    return y2d.view(*batch, n_cols), BLOCK_SIZE, num_warps, multi_block_launch


def _softmax_backward(
    dy: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE: int,
    num_warps: int,
    multi_block_launch: bool,
) -> torch.Tensor:
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)

    if not multi_block_launch and n_cols <= BLOCK_SIZE:
        _softmax_single_block_backward_kernel[(n_rows,)](
            dy2d,
            dy2d.stride(0),
            y2d,
            y2d.stride(0),
            dx2d,
            dx2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        _softmax_multi_block_backward_kernel[(n_rows,)](
            dy2d,
            dy2d.stride(0),
            y2d,
            y2d.stride(0),
            dx2d,
            dx2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return dx2d.view(*batch, n_cols)


class LigerSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor):
        y, BLOCK_SIZE, num_warps, multi_block_launch = _softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.multi_block_launch = multi_block_launch
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        dx = _softmax_backward(
            grad_output,
            y,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.multi_block_launch,
        )
        return dx


custom_softmax = LigerSoftmaxFunction.apply


def softmax_one(x):
    rescaled_input = x - torch.max(x, dim=-1, keepdim=True)[0]
    exp_input = torch.exp(rescaled_input)
    softmax = exp_input / (1 + torch.sum(exp_input, dim=-1, keepdim=True))
    return softmax


if __name__ == "__main__":
    from torch.nn.functional import softmax

    torch.manual_seed(0)
    sample = torch.randn((2, 4, 3), requires_grad=True)
    torch_ver = softmax(sample, dim=-1)
    custom_ver = softmax_one(sample)
    triton_ver = custom_softmax(sample.to("cuda"))
    print("Torch", torch_ver)
    print("Custom", custom_ver)
    print("Triton", triton_ver)
    print(torch.allclose(torch_ver, custom_ver.cpu()))
    print(torch.allclose(torch_ver, triton_ver.cpu()))

    torch_ver.sum().backward()
    torch_grads = sample.grad.clone()
    sample.grad.zero_()

    custom_ver.sum().backward()
    custom_grads = sample.grad.clone()
    sample.grad.zero_()

    triton_ver.sum().backward()
    triton_grads = sample.grad.clone()
    sample.grad.zero_()

    print("Torch", torch_grads)
    print("Custom", custom_grads)
    print("Triton", triton_grads)
    print(torch.allclose(torch_grads, custom_grads.cpu(), atol=1e-5))
    print(torch.allclose(torch_grads, triton_grads.cpu(), atol=1e-5))

    print("done")

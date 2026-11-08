"""GPU implementation, imported lazily only after input validation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul(A, B, C, M, N, K, A_BATCH, B_BATCH, C_BATCH,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Flatten output tiles into grid x; grid y selects an independent batch.
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_n = tl.cdiv(N, BN)
    rows = (tile // tiles_n) * BM + tl.arange(0, BM)
    cols = (tile % tiles_n) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + reduction
        a = tl.load(A + batch * A_BATCH + rows[:, None] * K + ks[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        b = tl.load(B + batch * B_BATCH + ks[:, None] * N + cols[None, :],
                    mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(C + batch * C_BATCH + rows[:, None] * N + cols[None, :],
             acc.to(tl.float16), mask=(rows[:, None] < M) & (cols[None, :] < N))


def launch(config, problem, a, b):
    shape = (problem.m, problem.n)
    if problem.op == "bmm":
        shape = (problem.batch, *shape)
    # Launch on the inputs' device even when another CUDA device is current.
    with torch.cuda.device(a.device):
        c = torch.empty(shape, device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(problem.m, config.block_m) * triton.cdiv(problem.n, config.block_n),
                problem.batch)
        _matmul[grid](a, b, c, problem.m, problem.n, problem.k,
                      problem.m * problem.k, problem.k * problem.n, problem.m * problem.n,
                      BM=config.block_m, BN=config.block_n, BK=config.block_k,
                      num_warps=config.num_warps, num_stages=config.num_stages)
    return c

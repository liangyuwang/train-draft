# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import time
from typing import Tuple

import torch

from model.ops.grouped_gemm import cg_grouped_gemm_forward


def create_aligned_test_data(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int,
    group_size_m: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data with proper block alignment.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension (K)
        output_dim: Output dimension (N)
        num_experts: Number of experts
        group_size_m: Size of expert groups
        device: Device to create tensors on
        dtype: Data type for inputs and weights

    Returns:
        Tuple of (inputs, expert_weights, expert_indices)
    """
    # Calculate total number of tokens
    M_total = batch_size * seq_len

    # Ensure M_total is a multiple of group_size_m
    padded_M = ((M_total + group_size_m - 1) // group_size_m) * group_size_m
    padding_needed = padded_M - M_total

    if padding_needed > 0:
        print(f"Padding input from {M_total} to {padded_M} to ensure group alignment")
        M_total = padded_M

    # Create inputs
    inputs = torch.randn((M_total, hidden_dim), dtype=dtype, device=device)

    # Create expert weights
    expert_weights = torch.randn(
        (num_experts, output_dim, hidden_dim), dtype=dtype, device=device
    )

    # Create expert indices with proper group alignment
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)

    # Assign experts in contiguous blocks of group_size_m
    num_groups = M_total // group_size_m

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size_m
        end_idx = start_idx + group_size_m

        # Assign this entire group to one expert
        expert_idx = group_idx % num_experts
        expert_indices[start_idx:end_idx] = expert_idx

    return inputs, expert_weights, expert_indices


def pytorch_reference(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Reference implementation using PyTorch for verification.
    """
    M_total, K = inputs.shape
    num_experts, N, _ = expert_weights.shape

    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Process each group
    for i in range(0, M_total, group_size_m):
        end_idx = min(i + group_size_m, M_total)

        # Get expert index for this group
        expert_idx = expert_indices[i].item()

        # Get expert weights
        expert_weight = expert_weights[expert_idx]

        # Compute output for this group
        output[i:end_idx] = torch.matmul(inputs[i:end_idx], expert_weight.t())

    return output


def verify_results(
    output_triton: torch.Tensor,
    output_reference: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """
    Verify that the Triton output matches the reference output.
    """
    is_close = torch.allclose(output_triton, output_reference, rtol=rtol, atol=atol)

    if not is_close:
        # Compute error statistics
        abs_diff = torch.abs(output_triton - output_reference)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()

        # Find location of maximum difference
        flat_idx = torch.argmax(abs_diff.view(-1))
        row = flat_idx // output_triton.shape[1]
        col = flat_idx % output_triton.shape[1]

        print("Results do not match!")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Max difference at [{row}, {col}]")
        print(f"Triton: {output_triton[row, col].item():.6f}")
        print(f"Reference: {output_reference[row, col].item():.6f}")

        return False

    return True


def run_shape_test(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int,
    group_size_m: int = 128,
    check_correctness: bool = True,
) -> bool:
    """Run a single shape test."""
    print(f"\nRunning test: batch={batch_size}, seq={seq_len}, "
          f"hidden={hidden_dim}, output={output_dim}, experts={num_experts}")

    # Memory check
    M_total = batch_size * seq_len
    est_memory = (
        M_total * hidden_dim     # inputs
        + num_experts * output_dim * hidden_dim  # weights
        + M_total * output_dim   # outputs
    ) * 2  # dtype=bfloat16
    est_memory_gb = est_memory / 1e9
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if est_memory_gb > 0.8 * total_mem:
        print(f"Skipping test - estimated {est_memory_gb:.2f} GB > 80% GPU memory")
        return True

    # Create test data
    inputs, expert_weights, expert_indices = create_aligned_test_data(
        batch_size, seq_len, hidden_dim, output_dim, num_experts, group_size_m
    )

    # Triton implementation
    print("Running Triton implementation...")
    output_triton = cg_grouped_gemm_forward(
        inputs, expert_weights, expert_indices, group_size_m=group_size_m
    )

    if check_correctness:
        # PyTorch reference
        print("Running PyTorch reference...")
        output_reference = pytorch_reference(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )

        # Verify results
        print("Verifying results...")
        is_correct = verify_results(output_triton, output_reference)
        print(f"Test {'passed' if is_correct else 'failed'}")
        return is_correct
    else:
        print("Skipping correctness check (benchmark mode).")
        return True


def benchmark_performance(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int,
    group_size_m: int = 128,
    num_runs: int = 10,
) -> bool:
    """Benchmark Triton vs PyTorch for a given shape."""

    print(
        f"\n[Benchmark] batch={batch_size}, seq={seq_len}, "
        f"hidden={hidden_dim}, output={output_dim}, experts={num_experts}"
    )

    # Create test data
    inputs, expert_weights, expert_indices = create_aligned_test_data(
        batch_size, seq_len, hidden_dim, output_dim, num_experts, group_size_m
    )

    # Warmup
    for _ in range(3):
        _ = cg_grouped_gemm_forward(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )
        _ = pytorch_reference(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )
        torch.cuda.synchronize()

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = cg_grouped_gemm_forward(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )
        torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = pytorch_reference(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # Compute FLOPs / TFLOPS
    M = inputs.shape[0]
    flops = 2 * M * hidden_dim * output_dim
    triton_tflops = flops / (triton_time / 1000) / 1e12
    pytorch_tflops = flops / (pytorch_time / 1000) / 1e12
    speedup = pytorch_time / triton_time

    # Print results
    print(f"  Triton: {triton_time:.2f} ms ({triton_tflops:.2f} TFLOPS)")
    print(f"  PyTorch: {pytorch_time:.2f} ms ({pytorch_tflops:.2f} TFLOPS)")
    print(f"  Speedup: {speedup:.2f}x")

    # Print summary table
    num_groups = M // group_size_m
    m_per_group = M / num_groups
    print(
        f"  table: {num_experts}\t{num_groups}\t{int(m_per_group)}\t"
        f"{hidden_dim}\t{output_dim}\t"
        f"{int(triton_tflops)} TFLOPS\t{int(triton_time)} ms\t{speedup:.1f}x"
    )

    return speedup > 0.9



def run_all_tests():
    """Run multiple shape tests with a wide variety of configurations."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return False

    # Define a variety of shapes to test
    shapes_to_test = [
        # ---- Small ----
        dict(batch_size=2, seq_len=16, hidden_dim=64, output_dim=64, num_experts=2),
        dict(batch_size=4, seq_len=32, hidden_dim=128, output_dim=128, num_experts=4),
        dict(batch_size=8, seq_len=64, hidden_dim=256, output_dim=256, num_experts=4),

        # ---- Medium ----
        dict(batch_size=16, seq_len=128, hidden_dim=512, output_dim=512, num_experts=4),
        dict(batch_size=16, seq_len=128, hidden_dim=1024, output_dim=1024, num_experts=8),
        dict(batch_size=32, seq_len=128, hidden_dim=1024, output_dim=2048, num_experts=8),
        dict(batch_size=32, seq_len=256, hidden_dim=2048, output_dim=2048, num_experts=8),
        dict(batch_size=32, seq_len=256, hidden_dim=2048, output_dim=4096, num_experts=16),

        # ---- Large ----
        dict(batch_size=32, seq_len=512, hidden_dim=4096, output_dim=4096, num_experts=8),
        dict(batch_size=32, seq_len=512, hidden_dim=4096, output_dim=7168, num_experts=8),
        dict(batch_size=32, seq_len=1024, hidden_dim=4096, output_dim=7168, num_experts=8),
        dict(batch_size=64, seq_len=1024, hidden_dim=4096, output_dim=7168, num_experts=16),

        # ---- Extra Large ----
        dict(batch_size=64, seq_len=2048, hidden_dim=4096, output_dim=4096, num_experts=16),
        dict(batch_size=64, seq_len=2048, hidden_dim=8192, output_dim=8192, num_experts=16),
        dict(batch_size=128, seq_len=1024, hidden_dim=8192, output_dim=8192, num_experts=32),

        # ---- Large num of Experts ----
        dict(batch_size=4, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=512),
        dict(batch_size=8, seq_len=2048, hidden_dim=2048, output_dim=1024, num_experts=256),
        dict(batch_size=8, seq_len=1024, hidden_dim=4096, output_dim=2048, num_experts=128),
    ]

    results = []
    for cfg in shapes_to_test:
        results.append(run_shape_test(**cfg))
        # Benchmark performance
        results.append(benchmark_performance(**cfg))

    all_passed = all(results)
    print(f"\nOverall test result: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    return all_passed


if __name__ == "__main__":
    run_all_tests()
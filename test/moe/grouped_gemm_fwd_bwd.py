import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import time
from typing import Tuple

import torch

from model.ops.grouped_gemm import cg_grouped_gemm
from grouped_gemm import create_aligned_test_data, pytorch_reference, verify_results


def run_forward_backward_test(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int,
    group_size_m: int = 128,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Test forward + backward of cg_grouped_gemm vs PyTorch reference."""

    print(
        f"\n[Forward+Backward Test] batch={batch_size}, seq={seq_len}, "
        f"hidden={hidden_dim}, output={output_dim}, experts={num_experts}"
    )

    # ===== 1. Create data =====
    inputs, expert_weights, expert_indices = create_aligned_test_data(
        batch_size, seq_len, hidden_dim, output_dim, num_experts, group_size_m
    )
    inputs.requires_grad_(True)
    expert_weights.requires_grad_(True)

    # ===== 2. Triton implementation =====
    output_triton = cg_grouped_gemm(
        inputs, expert_weights, expert_indices, group_size_m=group_size_m
    )
    loss_triton = output_triton.sum()
    loss_triton.backward()
    grad_inputs_triton = inputs.grad.detach().clone()
    grad_weights_triton = expert_weights.grad.detach().clone()

    # ===== 3. Reference implementation =====
    inputs_ref = inputs.detach().clone().requires_grad_(True)
    weights_ref = expert_weights.detach().clone().requires_grad_(True)

    output_ref = pytorch_reference(
        inputs_ref, weights_ref, expert_indices, group_size_m=group_size_m
    )
    loss_ref = output_ref.sum()
    loss_ref.backward()
    grad_inputs_ref = inputs_ref.grad.detach().clone()
    grad_weights_ref = weights_ref.grad.detach().clone()

    # ===== 4. Compare =====
    forward_ok = verify_results(output_triton, output_ref, rtol=rtol, atol=atol)
    if not forward_ok:
        print("❌ Forward mismatch!")
    grad_inputs_ok = verify_results(
        grad_inputs_triton, grad_inputs_ref, rtol=rtol, atol=atol
    )
    if not grad_inputs_ok:
        print("❌ Input gradient mismatch!")
    grad_weights_ok = verify_results(
        grad_weights_triton, grad_weights_ref, rtol=rtol, atol=atol
    )
    if not grad_weights_ok:
        print("❌ Weight gradient mismatch!")

    all_ok = forward_ok and grad_inputs_ok and grad_weights_ok
    print(f"Forward+Backward test {'passed ✅' if all_ok else 'failed ❌'}")

    return all_ok


def benchmark_performance(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int,
    group_size_m: int = 128,
    num_runs: int = 10,
) -> bool:
    """Benchmark forward+backward of cg_grouped_gemm vs PyTorch reference."""

    print(
        f"\n[Benchmark FWD+BWD] batch={batch_size}, seq={seq_len}, "
        f"hidden={hidden_dim}, output={output_dim}, experts={num_experts}"
    )

    # ===== 1. Create data =====
    inputs, expert_weights, expert_indices = create_aligned_test_data(
        batch_size, seq_len, hidden_dim, output_dim, num_experts, group_size_m
    )

    # ===== 2. Warmup =====
    for _ in range(3):
        x = inputs.detach().clone().requires_grad_(True)
        w = expert_weights.detach().clone().requires_grad_(True)
        y = cg_grouped_gemm(x, w, expert_indices, group_size_m=group_size_m)
        y.sum().backward()
        torch.cuda.synchronize()

        x_ref = inputs.detach().clone().requires_grad_(True)
        w_ref = expert_weights.detach().clone().requires_grad_(True)
        y_ref = pytorch_reference(x_ref, w_ref, expert_indices, group_size_m=group_size_m)
        y_ref.sum().backward()
        torch.cuda.synchronize()

    # ===== 3. Benchmark Triton =====
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        x = inputs.detach().clone().requires_grad_(True)
        w = expert_weights.detach().clone().requires_grad_(True)
        y = cg_grouped_gemm(x, w, expert_indices, group_size_m=group_size_m)
        y.sum().backward()
        torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_runs * 1000  # ms

    # ===== 4. Benchmark PyTorch =====
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        x_ref = inputs.detach().clone().requires_grad_(True)
        w_ref = expert_weights.detach().clone().requires_grad_(True)
        y_ref = pytorch_reference(x_ref, w_ref, expert_indices, group_size_m=group_size_m)
        y_ref.sum().backward()
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # ===== 5. Compute TFLOPS =====
    M = inputs.shape[0]  # 注意这里用 pad 后的 token 数
    fwd_flops = 2 * M * hidden_dim * output_dim
    bwd_flops = 2 * fwd_flops  # 大约 forward 的两倍（input grad + weight grad）
    total_flops = fwd_flops + bwd_flops

    triton_tflops = total_flops / (triton_time / 1000) / 1e12
    pytorch_tflops = total_flops / (pytorch_time / 1000) / 1e12
    speedup = pytorch_time / triton_time

    # ===== 6. Report =====
    print(f"  Triton: {triton_time:.2f} ms ({triton_tflops:.2f} TFLOPS)")
    print(f"  PyTorch: {pytorch_time:.2f} ms ({pytorch_tflops:.2f} TFLOPS)")
    print(f"  Speedup: {speedup:.2f}x")

    num_groups = M // group_size_m if M >= group_size_m else 1
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
        dict(batch_size=8, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=2),
        dict(batch_size=8, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=4),
        dict(batch_size=8, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=8),

        # ---- Medium ----
        dict(batch_size=8, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=16),
        dict(batch_size=8, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=32),
        dict(batch_size=8, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=64),

        # ---- Large ----
        dict(batch_size=4, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=128),
        dict(batch_size=4, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=256),
        dict(batch_size=4, seq_len=4096, hidden_dim=1024, output_dim=768, num_experts=512),
    ]

    results = []
    for cfg in shapes_to_test:
        results.append(run_forward_backward_test(**cfg))
        # Benchmark performance
        results.append(benchmark_performance(**cfg))

    all_passed = all(results)
    print(f"\nOverall test result: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    return all_passed


if __name__ == "__main__":
    run_all_tests()
"""
MMA Atom Visualization Example

This example demonstrates visualizing MMA (Matrix Multiply-Accumulate) atoms
using the high-level API. It shows how threads are distributed across the
A, B, and C matrices in a matrix multiplication operation using Tensor Cores.

The visualization shows the standard layout:
    B
A   C

Where C = A × B for matrix multiplication with a 16×8×8 tile (FP16 Tensor Core).
"""

from cutlass import cute, Float16, Float32
from cute_viz import render_tiled_mma_svg, display_tiled_mma


@cute.jit
def main():
    print("MMA Atom Visualization Example")
    print("=" * 60)

    # Define tile size: M=16, N=8, K=8 (A100+ Tensor Core shape)
    tile_mnk = (16, 8, 8)
    M, N, K = tile_mnk

    print(f"Tile dimensions: M={M}, N={N}, K={K}")
    print()

    # Create MMA operation for Float16 Tensor Cores
    # This uses the native HMMA.16816 instruction
    op = cute.nvgpu.warp.MmaF16BF16Op(
        Float16,      # Input A/B type (FP16)
        Float32,      # Accumulator type (FP32)
        (16, 8, 8)    # MMA shape: M×N×K
    )

    print(f"MMA Operation: {op}")
    print()

    print("Using 16×8×8 Tensor Core operation (SM80+)")
    print("  • Uses full warp (32 threads)")
    print("  • Input: FP16, Output: FP32")
    print("  • Native hardware instruction")
    print()

    # Create the TiledMMA with just the operation
    # The operation defines its own thread layout (32 threads in a warp)
    tiled_mma = cute.make_tiled_mma(op)

    print("TiledMMA created successfully!")
    print()

    # Show TiledMMA properties
    print("TiledMMA properties:")
    print(f"  TiledMMA: {tiled_mma}")
    print()

    # Visualize with the high-level API
    print("Generating visualization...")
    render_tiled_mma_svg(tiled_mma, tile_mnk, "assets/mma_layout.svg")
    print("✓ MMA layout saved to assets/mma_layout.svg")
    print()

    print("The visualization shows:")
    print("  • A (left): How threads access matrix A (M×K)")
    print("  • B (top): How threads access matrix B (N×K, shown transposed)")
    print("  • C (bottom-right): How threads compute/store matrix C (M×N)")
    print("  • Layout follows matrix multiplication: C = A × B")
    print("  • Colors indicate thread IDs (same color = same thread)")
    print("  • Numbers show value IDs within each thread")
    print()
    print("This demonstrates the thread-value mapping for MMA operations,")
    print("showing how a single instruction distributes work across threads.")

    # For Jupyter notebooks, you can use:
    # display_tiled_mma(tiled_mma, tile_mnk)


if __name__ == "__main__":
    main()

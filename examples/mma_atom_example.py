"""
MMA Atom Visualization Example

Demonstrates MMA (Matrix Multiply-Accumulate) layout visualization using
SM80+ native Tensor Core operations (16×8×8).
"""

from cutlass import cute, Float16, Float32
from cute_viz import render_tiled_mma_svg


@cute.jit
def main():
    print("SM80 16×8×8 MMA Atom Visualization")
    print("=" * 60)

    # Tile dimensions: M×N×K = 16×8×8
    tile_mnk = (16, 8, 8)

    # Create MMA atom using native SM80+ Tensor Core instruction
    # MmaF16BF16Op: F16/BF16 input, F32 accumulator, shape (16,8,8)
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, tile_mnk)

    # Create TiledMMA
    tiled_mma = cute.make_tiled_mma(mma_atom)

    print(f"Tile MNK: {tile_mnk}")
    print()

    # Render MMA layout visualization
    print("Generating MMA layout visualization...")
    render_tiled_mma_svg(tiled_mma, tile_mnk, "assets/mma_layout.svg")
    print("✓ MMA layout saved to assets/mma_layout.svg")
    print()

    print("This visualization shows:")
    print("  - Matrix C (M×N): Accumulator matrix at bottom-right")
    print("  - Matrix A (M×K): Left input matrix")
    print("  - Matrix B (N×K): Top input matrix (transposed view)")
    print("  - Thread IDs (T0-T31): Which thread owns each element")
    print("  - Value IDs (V0-V3): Which register within each thread")


if __name__ == "__main__":
    main()

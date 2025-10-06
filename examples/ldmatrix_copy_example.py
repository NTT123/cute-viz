"""
LDMATRIX Copy Atom Visualization Example

This example demonstrates visualizing ldmatrix copy atoms, which wrap
the CUDA ldmatrix instruction for efficient warp-level matrix loads
from shared memory.

The ldmatrix instruction is specifically designed to solve memory access
performance issues when loading data for Tensor Core operations.

Available ldmatrix copy atoms in Python:
- LdMatrix8x8x16bOp: Load 8x8 matrix of 16-bit elements
- LdMatrix16x16x8bOp: Load 16x16 matrix of 8-bit elements
"""

from cutlass import cute, Float16
from cute_viz import render_tiled_copy_svg


@cute.jit
def main():
    print("LDMATRIX Copy Atom Visualization")
    print("=" * 60)

    # Create ldmatrix copy atom for 16-bit (half-precision) data
    # LdMatrix8x8x16bOp loads an 8x8 matrix of 16-bit elements
    # This is the Python equivalent of SM75_U16x8_LDSM_T
    ldmatrix_op = cute.nvgpu.warp.LdMatrix8x8x16bOp()
    copy_atom = cute.make_copy_atom(ldmatrix_op, Float16)

    print("Copy Atom: LdMatrix8x8x16bOp (ldmatrix for 16-bit data)")
    print(f"  Operation: {ldmatrix_op}")
    print(f"  Copy atom: {copy_atom}")
    print()

    # The copy atom has specific characteristics
    print("Copy atom characteristics:")
    print("  • Threads: 32 (one full warp)")
    print("  • Loads an 8x8 matrix of 16-bit elements per operation")
    print("  • Each thread loads 2 elements (64 total / 32 threads)")
    print("  • Optimized for coalesced memory access and Tensor Core layout")
    print()

    # For ldmatrix 8x8x16b, the natural tile size is 8x8
    # The atom loads 64 elements (8×8) using 32 threads (2 elements/thread)
    tile_mn = (8, 8)

    # Create thread and value layouts for the tiled copy
    # For a single 8x8 ldmatrix operation:
    # - 32 threads in the warp
    # - Each thread handles 2 elements from the 8x8 matrix

    # Simple configuration: use the natural layout
    thr_layout = cute.make_layout((32, 1))
    val_layout = cute.make_layout((1, 2))

    # Create the tiled copy
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    print(f"Tiled copy for {tile_mn} tile:")
    print(f"  • Thread layout: {thr_layout}")
    print(f"  • Value layout: {val_layout}")
    print()

    # Show TiledCopy properties
    print("TiledCopy properties:")
    print(f"  • layout_src_tv_tiled: {tiled_copy.layout_src_tv_tiled}")
    print(f"  • layout_dst_tv_tiled: {tiled_copy.layout_dst_tv_tiled}")
    print(f"  • size: {tiled_copy.size}")
    print()

    # Visualize the copy pattern
    print("Generating visualization...")
    render_tiled_copy_svg(tiled_copy, tile_mn, "assets/ldmatrix_copy.svg")
    print("✓ LDMATRIX copy layout saved to assets/ldmatrix_copy.svg")
    print()

    print("The visualization shows:")
    print("  • Source layout: Memory access pattern from shared memory")
    print("  • Destination layout: Register arrangement for Tensor Cores")
    print("  • Colors indicate which thread owns each element")
    print("  • The pattern optimizes for coalesced memory access")


if __name__ == "__main__":
    main()

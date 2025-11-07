"""
Example demonstrating hierarchical layout visualization.

This example creates and visualizes various hierarchical CuTe layouts,
including layouts with different nesting patterns.
"""

from cutlass import cute, range_constexpr
from cute_viz import render_layout_svg


@cute.jit
def demonstrate_hierarchical_layouts():
    """Demonstrate various hierarchical layout patterns."""
    
    print("=" * 60)
    print("EXAMPLE 1: Row-major layout (2, 4):(4, 1)")
    print("=" * 60)
    layout1 = cute.make_layout((2, 4), stride=(4, 1))
    print(f"Layout: {layout1}")
    print(f"Rank: {cute.rank(layout1)}")
    print(f"Size: {cute.size(layout1)}")
    
    print("\nMapping (row-major):")
    for i in range_constexpr(2):
        for j in range_constexpr(4):
            idx = layout1((i, j))
            print(f"({i},{j})->{idx}", end="  ")
        print()
    
    render_layout_svg(layout1, "assets/row_major_layout.svg")
    print("✓ Saved to assets/row_major_layout.svg\n")
    
    print("=" * 60)
    print("EXAMPLE 2: Hierarchical layout (2,(2,2)):(4,(2,1))")
    print("=" * 60)
    layout2 = cute.make_layout((2, (2, 2)), stride=(4, (2, 1)))
    print(f"Layout: {layout2}")
    print(f"Rank: {cute.rank(layout2)}")
    print(f"Size: {cute.size(layout2)}")
    print(f"Depth mode 0: {cute.depth(layout2[0])}")
    print(f"Depth mode 1: {cute.depth(layout2[1])}")
    
    print("\nHierarchical structure: 2 tiles of 2×2 grids")
    print("\nMapping:")
    for i in range_constexpr(2):
        print(f"Tile {i}:")
        for j in range_constexpr(2):
            for k in range_constexpr(2):
                idx = layout2((i, (j, k)))
                print(f"  ({i},({j},{k}))->{idx}", end="")
            print()
    
    render_layout_svg(layout2, "assets/hierarchical_layout_2_2x2.svg")
    print("✓ Saved to assets/hierarchical_layout_2_2x2.svg\n")
    
    print("=" * 60)
    print("EXAMPLE 3: Column-major layout (2, 4):(1, 2)")
    print("=" * 60)
    layout3 = cute.make_layout((2, 4))
    print(f"Layout: {layout3}")
    
    print("\nMapping (column-major):")
    for i in range_constexpr(2):
        for j in range_constexpr(4):
            idx = layout3((i, j))
            print(f"({i},{j})->{idx}", end="  ")
        print()
    
    render_layout_svg(layout3, "assets/col_major_layout.svg")
    print("✓ Saved to assets/col_major_layout.svg\n")
    
    print("=" * 60)
    print("EXAMPLE 4: Hierarchical layout ((2, 2), 2)")
    print("=" * 60)
    layout4 = cute.make_layout(((2, 2), 2))
    print(f"Layout: {layout4}")
    print(f"Rank: {cute.rank(layout4)}")
    print(f"Size: {cute.size(layout4)}")
    print(f"Depth mode 0: {cute.depth(layout4[0])}")
    print(f"Depth mode 1: {cute.depth(layout4[1])}")
    
    print("\nHierarchical structure: 2 tiles, each 2×2")
    print("\nMapping:")
    for i in range_constexpr(2):
        for j in range_constexpr(2):
            for k in range_constexpr(2):
                idx = layout4(((i, j), k))
                print(f"(({i},{j}),{k})->{idx}", end="  ")
        print()
    
    render_layout_svg(layout4, "assets/hierarchical_layout_2x2_2.svg")
    print("✓ Saved to assets/hierarchical_layout_2x2_2.svg\n")
    
    print("=" * 60)
    print("EXAMPLE 5: Hierarchical layout ((2,2), (3, 4))")
    print("=" * 60)
    layout5 = cute.make_layout(((2, 2), (3, 4)))
    print(f"Layout: {layout5}")
    print(f"Rank: {cute.rank(layout5)}")
    print(f"Size: {cute.size(layout5)}")
    print(f"Depth mode 0: {cute.depth(layout5[0])}")
    print(f"Depth mode 1: {cute.depth(layout5[1])}")
    
    print("\nHierarchical structure: 3×4 grid of tiles, each tile is 2×2")
    print("\nMapping (first few elements):")
    for i in range_constexpr(2):
        for j in range_constexpr(2):
            print(f"Row ({i},{j}):")
            for k in range_constexpr(3):
                print(" ", end="")
                for l in range_constexpr(4):
                    idx = layout5(((i, j), (k, l)))
                    print(f"(({i},{j}),({k},{l}))->{idx:2d}", end=" ")
                print()
    
    render_layout_svg(layout5, "assets/hierarchical_layout_2x2_3x4.svg")
    print("✓ Saved to assets/hierarchical_layout_2x2_3x4.svg\n")
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_hierarchical_layouts()


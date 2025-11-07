"""
Example demonstrating 1D layout visualization.

This example creates and visualizes simple 1D CuTe layouts.
"""

from cutlass import cute
from cute_viz import render_layout_svg, display_layout


@cute.jit
def main():
    # Create a simple 1D layout with 8 elements
    layout_8 = cute.make_layout(8)
    print(f"1D Layout (8 elements): {layout_8}")
    render_layout_svg(layout_8, "assets/1d_layout_8.svg")
    
    # Create a 1D layout with custom stride
    layout_stride = cute.make_layout(8, stride=2)
    print(f"1D Layout (stride=2): {layout_stride}")
    render_layout_svg(layout_stride, "assets/1d_layout_stride2.svg")
    
    # Create a longer 1D layout
    layout_16 = cute.make_layout(16)
    print(f"1D Layout (16 elements): {layout_16}")
    render_layout_svg(layout_16, "assets/1d_layout_16.svg")
    
    print("\nSVG files created successfully in assets/!")
    print("- assets/1d_layout_8.svg")
    print("- assets/1d_layout_stride2.svg")
    print("- assets/1d_layout_16.svg")


if __name__ == "__main__":
    main()


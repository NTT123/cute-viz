"""
Thread-Value Layout Visualization Example

This example demonstrates how to visualize a CuTe thread-value (TV) layout.
"""

from cutlass import cute
from cute_viz import render_tv_layout_svg, display_tv_layout


@cute.jit
def main():
    # Create a thread-value layout
    tile_mn = (8, 8)
    tv_layout = cute.make_layout(
        shape=((2, 2, 2), (2, 2, 2)),
        stride=((1, 16, 4), (8, 2, 32))
    )

    # Render to SVG file
    render_tv_layout_svg(tv_layout, tile_mn, "assets/tv_layout.svg")
    print("TV layout saved to assets/tv_layout.svg")

    # Or display directly in Jupyter notebook
    # Uncomment the line below when running in Jupyter
    # display_tv_layout(tv_layout, tile_mn)


if __name__ == "__main__":
    main()

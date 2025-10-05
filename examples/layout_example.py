"""
Basic Layout Visualization Example

This example demonstrates how to visualize a CuTe layout.
"""

from cutlass import cute
from cute_viz import render_layout_svg, display_layout


@cute.jit
def main():
    # Create a layout with shape (4, 6) and stride (3, 1)
    layout = cute.make_layout((4, 6), stride=(3, 1))

    # Render to SVG file
    render_layout_svg(layout, "layout.svg")
    print("Layout saved to layout.svg")

    # Or display directly in Jupyter notebook
    # Uncomment the line below when running in Jupyter
    # display_layout(layout)


if __name__ == "__main__":
    main()

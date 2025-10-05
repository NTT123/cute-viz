"""
Swizzle Layout Visualization Example

This example demonstrates how to create and visualize a CuTe Swizzle layout.

A Swizzle is a transformation that permutes elements to improve memory access patterns.
It's defined by three parameters:
- MBase: Number of least-significant bits to keep constant
- BBits: Number of bits in the mask
- SShift: Distance to shift the mask
"""

from cutlass import cute
from cute_viz import render_swizzle_layout_svg, display_swizzle_layout


@cute.jit
def main():
    # Create a basic layout (8x8 for clear visualization)
    base_layout = cute.make_layout((8, 8), stride=(8, 1))

    print("Base layout:")
    print(base_layout)

    # Create a Swizzle transformation
    # Example: Swizzle<3, 0, 3> - shows clear swizzling pattern
    # Parameters for make_swizzle(b, m, s):
    # b (BBits)=3: Number of bits in the mask (8 possible values)
    # m (MBase)=0: Keep 0 least-significant bits constant
    # s (SShift)=3: Shift the mask by 3 positions
    # This swizzle XORs bits to rearrange elements for better memory access
    swizzle = cute.make_swizzle(b=3, m=0, s=3)

    print("\nSwizzle:")
    print(swizzle)

    # Apply swizzle to the layout using make_composed_layout
    # The composition is (swizzle ∘ offset ∘ base_layout)
    # Use offset=0 for simple swizzling without translation
    swizzled_layout = cute.make_composed_layout(swizzle, 0, base_layout)

    print("\nSwizzled layout:")
    print(swizzled_layout)

    # Render to SVG file
    render_swizzle_layout_svg(swizzled_layout, "assets/swizzle_layout.svg")
    print("\nSwizzle layout saved to assets/swizzle_layout.svg")

    # Or display directly in Jupyter notebook
    # Uncomment the line below when running in Jupyter
    # display_swizzle_layout(swizzled_layout)


if __name__ == "__main__":
    main()

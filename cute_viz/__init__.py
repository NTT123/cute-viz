"""
cute_viz: Visualization package for CuTe layouts

This package provides functions to visualize CuTe tensor layouts as SVG images.
"""

from .core import (
    render_layout_svg,
    render_tv_layout_svg,
    render_swizzle_layout_svg,
    render_copy_layout_svg,
    render_tiled_copy_svg,
    display_svg,
    display_layout,
    display_tv_layout,
    display_swizzle_layout,
    display_copy_layout,
    display_tiled_copy,
    tidfrg_S,
    tidfrg_D,
)

__version__ = "0.1.0"
__all__ = [
    "render_layout_svg",
    "render_tv_layout_svg",
    "render_swizzle_layout_svg",
    "render_copy_layout_svg",
    "render_tiled_copy_svg",
    "display_svg",
    "display_layout",
    "display_tv_layout",
    "display_swizzle_layout",
    "display_copy_layout",
    "display_tiled_copy",
    "tidfrg_S",
    "tidfrg_D",
]
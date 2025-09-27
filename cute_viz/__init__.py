"""
cute_viz: Visualization package for CuTe layouts

This package provides functions to visualize CuTe tensor layouts as SVG images.
"""

from .core import render_layout_svg, render_tv_layout_svg, display_svg

__version__ = "0.1.0"
__all__ = ["render_layout_svg", "render_tv_layout_svg", "display_svg"]
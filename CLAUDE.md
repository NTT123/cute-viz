# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is "cute-viz", a Python package for visualizing CuTe tensor layouts from NVIDIA's CUTLASS library. The package provides functions to render CuTe layouts as SVG images for better understanding of GPU tensor memory layouts and thread mappings.

## Development Commands

The project uses a standard Python package structure with pyproject.toml configuration:

- **Install in development mode**: `pip install -e .`
- **Install from GitHub**: `pip install git+https://github.com/NTT123/cute-viz.git`
- **Test the package**: Create a test script importing `cute_viz` functions

## Project Structure

- `cute_viz/` - Main package directory
  - `__init__.py` - Package initialization with public API exports
  - `core.py` - Core visualization functions
- `pyproject.toml` - Project configuration with dependencies
- `README.md` - Comprehensive package documentation
- `.python-version` - Python version specification (>=3.8)

## Dependencies

The package requires:
- numpy - Array operations
- nvidia-cutlass-dsl==4.2.0 - CUTLASS CuTe library
- cuda-bindings==12.9.1 - CUDA bindings
- cuda-python==12.9.1 - CUDA Python support
- svgwrite - SVG generation
- ipython - Jupyter notebook display support

## Architecture Notes

The package is structured as a Python library with:
- **Public API**: Three main functions exported from `cute_viz` module
- **render_layout_svg**: Visualizes basic CuTe layouts as color-coded grids
- **render_tv_layout_svg**: Visualizes thread-value layouts with thread/value IDs
- **display_svg**: Helper for displaying SVGs in Jupyter notebooks

The visualization functions generate SVG files with configurable cell sizes, color schemes, and labeling to help developers understand complex GPU tensor memory layouts.
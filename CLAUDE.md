# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is "cute-viz", a Python package for visualizing CuTe tensor layouts from NVIDIA's CUTLASS library. The package provides functions to render CuTe layouts as SVG images for better understanding of GPU tensor memory layouts and thread mappings.

## Development Commands

The project uses a standard Python package structure with pyproject.toml configuration:

- **Install in development mode**: `pip install -e .`
- **Install from GitHub**: `pip install git+https://github.com/NTT123/cute-viz.git`
- **Run examples**: `python examples/layout_example.py` or `python examples/mma_atom_example.py`

## Project Structure

- `cute_viz/` - Main package directory
  - `__init__.py` - Package initialization, exports public API:
    - Visualization: `render_layout_svg`, `render_tv_layout_svg`, `render_swizzle_layout_svg`, `render_copy_layout_svg`, `render_tiled_copy_svg`, `render_mma_layout_svg`, `render_tiled_mma_svg`
    - Display: `display_svg`, `display_layout`, `display_tv_layout`, `display_swizzle_layout`, `display_copy_layout`, `display_tiled_copy`, `display_mma_layout`, `display_tiled_mma`
    - Utilities: `tidfrg_S`, `tidfrg_D`
  - `core.py` - Core visualization functions with internal helpers and utility functions
- `examples/` - Runnable example scripts demonstrating package usage
- `pyproject.toml` - Project configuration with dependencies
- `README.md` - Package documentation with usage examples
- `.python-version` - Python version specification (>=3.12)

## Dependencies

The package requires:
- numpy - Array operations
- nvidia-cutlass-dsl>=4.2.0 - CUTLASS CuTe library
- cuda-bindings>=12.9.1 - CUDA bindings
- cuda-python>=12.9.1 - CUDA Python support
- svgwrite - SVG generation
- ipython - Jupyter notebook display support

## Architecture Notes

The package provides four types of layout visualizations:

1. **Basic Layout Visualization** (`render_layout_svg`, `display_layout`):
   - Visualizes CuTe layouts as color-coded grids with grayscale colors
   - Shows the linear index mapping for 2D tensor coordinates
   - Cell size: 20px, supports up to 8 distinct grayscale shades for different indices

2. **Thread-Value (TV) Layout Visualization** (`render_tv_layout_svg`, `display_tv_layout`):
   - Visualizes rank-2 TV layouts showing thread-to-memory mappings
   - Requires both a layout and a tile_mn parameter (rank-2 MN Tile)
   - Uses composition with identity tensor to map coordinates
   - Color-codes threads with 8 distinct pastel colors
   - Labels each cell with thread ID (T#) and value ID (V#)

3. **Swizzle Layout Visualization** (`render_swizzle_layout_svg`, `display_swizzle_layout`):
   - Visualizes swizzled layouts that permute elements for better memory access patterns
   - Uses the same grayscale color scheme as basic layouts
   - Internally attempts to convert to position-independent form if available
   - Shows the permuted memory access pattern created by the swizzle transformation

4. **Copy Layout Visualization** (`render_copy_layout_svg`, `display_copy_layout`):
   - Visualizes copy operations with source and destination TV layouts side-by-side
   - Shows how threads map to both source and destination memory locations
   - Two grids displayed horizontally with 3-cell gap between them
   - Uses same pastel color scheme as TV layouts (8 colors by thread ID)
   - Labels cells with thread ID (T#) and value ID (V#)
   - Useful for understanding data movement patterns in GPU memory copies

All visualization functions use SVG format with:
- 20px cell size (hardcoded)
- Black cell borders
- 8px font size for labels
- Color cycling based on modulo of index/thread count

The package supports both file-based output (render functions) and direct Jupyter notebook display (display functions). All CuTe operations must be wrapped in `@cute.jit` decorator.

## Python CuTe DSL Critical Notes

**IMPORTANT: Python CuTe DSL differs significantly from C++ CuTe API.**

### Creating Swizzle Layouts

**❌ WRONG - Don't use class constructors directly:**
```python
swizzle = cute.Swizzle(2, 3, 3)  # TypeError! Constructor expects MLIR IR Value
```

**✅ CORRECT - Use factory functions:**
```python
swizzle = cute.make_swizzle(b=2, m=3, s=3)  # Factory function creates proper MLIR representation
```

### Composing Swizzles with Layouts

**❌ WRONG - Using composition() hangs/fails:**
```python
swizzled_layout = cute.composition(base_layout, swizzle)  # Will hang indefinitely!
```

**✅ CORRECT - Use make_composed_layout:**
```python
# Creates composition: swizzle ∘ offset ∘ base_layout
swizzled_layout = cute.make_composed_layout(swizzle, 0, base_layout)
```

**Key Differences:**
- `composition(lhs, rhs)`: General-purpose layout-to-layout composition, computes `lhs(rhs(c))`
- `make_composed_layout(inner, offset, outer)`: Specifically designed for transformations like swizzles, creates `inner ∘ offset ∘ outer`
- **For swizzles, you MUST use `make_composed_layout()`**, not `composition()`

### Swizzle Parameters

Swizzles are defined by three parameters (b, m, s):
- **b (BBits)**: Number of bits in the mask
- **m (MBase)**: Number of least-significant bits to keep constant
- **s (SShift)**: Distance to shift the mask

Example: `cute.make_swizzle(b=2, m=3, s=3)` creates Swizzle<2,3,3> for memory bank conflict avoidance.

### Creating Copy Atoms and Types

**❌ WRONG - Using numpy types directly:**
```python
from cute_viz import render_copy_layout_svg
import numpy as np

copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), np.float32)  # AttributeError!
```

**✅ CORRECT - Use CuTe type objects:**
```python
from cutlass import cute, Float32

copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)  # Works correctly
```

**Available CuTe Types:**
- `Float32`, `Float16`, `BFloat16` - Floating point types
- `Int32`, `Int16`, `Int8` - Signed integer types
- `UInt32`, `UInt16`, `UInt8` - Unsigned integer types
- `Boolean` - Boolean type

These types have the `.mlir_type` attribute required by the JIT compiler.

### TiledCopy Visualization API

**High-Level API (Recommended):**

Python equivalent of C++ `print_latex(TiledCopy)` - visualize a TiledCopy with a single function call:

```python
from cutlass import cute, Float32
from cute_viz import render_tiled_copy_svg

@cute.jit
def main():
    tile_mn = (8, 8)

    # Create copy atom and tiled copy
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
    thr_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    val_layout = cute.make_ordered_layout((2, 2), order=(1, 0))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    # Visualize with one function call!
    render_tiled_copy_svg(tiled_copy, tile_mn, "copy_layout.svg")
```

For Jupyter notebooks, use `display_tiled_copy(tiled_copy, tile_mn)`.

**Low-Level API (Advanced):**

For fine-grained control, use `tidfrg_S` and `tidfrg_D` to extract thread-value layouts:

```python
from cute_viz import tidfrg_S, tidfrg_D, render_copy_layout_svg

# Extract layouts manually
layout_s_tv = tidfrg_S(tiled_copy, tile_mn)
layout_d_tv = tidfrg_D(tiled_copy, tile_mn)

# Custom processing of layouts here...

# Render
render_copy_layout_svg(layout_s_tv, layout_d_tv, tile_mn, "copy_layout.svg")
```

**C++ API Reference:**

In C++ CuTe, `TiledCopy` provides these methods:
- `tidfrg_S(tensor)` - Tiles source tensor into (thread, value) partitions
- `tidfrg_D(tensor)` - Tiles destination tensor into (thread, value) partitions
- `print_latex(TiledCopy)` - Renders TiledCopy visualization

**Implementation Details:**

The high-level API (`render_tiled_copy_svg`) automatically:
1. Calls `tidfrg_S` and `tidfrg_D` to extract source/destination layouts
2. Handles multi-dimensional tensor slicing (equivalent to C++ `(_,_,Int<0>{})`)
3. Renders side-by-side visualization of source and destination mappings

The low-level functions work by:
1. Creating an identity tensor with the tile shape: `make_identity_tensor(tile_mn)`
2. Composing it with the tiled copy's layouts: `composition(identity, tiled_copy.layout_src_tv_tiled)`

**Benefits:**
- ✅ API parity with C++ CuTe `print_latex(TiledCopy)`
- ✅ One-line visualization of any TiledCopy object
- ✅ Automatically extracts source and destination thread-value mappings
- ✅ Works with all copy atoms (CopyUniversalOp, cp.async, TMA, etc.)

### TiledMMA Visualization API

**High-Level API (Recommended):**

Python equivalent of C++ `print_latex(TiledMMA)` - visualize a TiledMMA with a single function call:

```python
from cutlass import cute, Float16, Float32
from cute_viz import render_tiled_mma_svg

@cute.jit
def main():
    # Tile dimensions: M×N×K = 16×8×8 (SM80+ Tensor Core)
    tile_mnk = (16, 8, 8)

    # Create MMA atom using native SM80+ Tensor Core instruction
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, tile_mnk)

    # Create TiledMMA
    tiled_mma = cute.make_tiled_mma(mma_atom)

    # Visualize with one function call!
    render_tiled_mma_svg(tiled_mma, tile_mnk, "mma_layout.svg")
```

For Jupyter notebooks, use `display_tiled_mma(tiled_mma, tile_mnk)`.

**Low-Level API (Advanced):**

The high-level API (`render_tiled_mma_svg` and `display_tiled_mma`) handles everything automatically. For advanced use cases, you can access the TiledMMA properties directly:

- `tiled_mma.tv_layout_C_tiled` - Thread-value layout for C matrix
- `tiled_mma.tv_layout_A_tiled` - Thread-value layout for A matrix
- `tiled_mma.tv_layout_B_tiled` - Thread-value layout for B matrix

**C++ API Reference:**

In C++ CuTe, `TiledMMA` provides these methods:
- `get_layoutC_TV()` - Thread-value layout for C matrix (M×N)
- `get_layoutA_TV()` - Thread-value layout for A matrix (M×K)
- `get_layoutB_TV()` - Thread-value layout for B matrix (N×K)
- `print_latex(TiledMMA)` - Renders TiledMMA visualization

**Python Property Names:**

In Python CuTe DSL, use properties instead of methods:
- `tiled_mma.tv_layout_C_tiled` (NOT `get_layoutC_TV()`)
- `tiled_mma.tv_layout_A_tiled` (NOT `get_layoutA_TV()`)
- `tiled_mma.tv_layout_B_tiled` (NOT `get_layoutB_TV()`)

**Visualization Layout:**

The MMA visualization shows the standard matrix multiplication layout:
```
    B (K×N, transposed)
A   C
```

Where C = A × B with dimensions:
- A: M×K (left)
- B: N×K shown transposed as K×N (top)
- C: M×N (bottom-right)

Colors indicate thread IDs, numbers show value IDs within each thread.

**Implementation Details:**

The high-level API (`render_tiled_mma_svg`) automatically:
1. Extracts TV layouts using `tv_layout_A_tiled`, `tv_layout_B_tiled`, `tv_layout_C_tiled` properties
2. Creates identity tensors for each matrix dimension
3. Composes layouts with identity tensors to get coordinate mappings
4. Handles multi-dimensional tensor slicing (equivalent to C++ `[:, :, 0]`)
5. Renders the three-panel MMA layout visualization

**Supported MMA Configurations:**

The Python CUTLASS DSL supports native MMA operations:
- **SM80+ (A100/H100)**: Native `cute.nvgpu.warp.MmaF16BF16Op` supports (16,8,8) and (16,8,16)
- **SM90+ (Hopper)**: Native `cute.nvgpu.tcgen05.MmaF16BF16Op` and warpgroup operations

The package example demonstrates SM80 16×8×8 using native Tensor Core operations.

**Benefits:**
- ✅ API parity with C++ CuTe `print_latex(TiledMMA)`
- ✅ One-line visualization of any TiledMMA object
- ✅ Automatically extracts A, B, C thread-value mappings
- ✅ Shows matrix multiplication structure visually
- ✅ Works with all MMA atoms (MmaUniversalOp, Tensor Core ops, etc.)

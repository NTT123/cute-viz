"""
Core visualization functions for CuTe layouts.
"""

import numpy as np
import svgwrite
from cutlass import cute, range_constexpr
from cutlass.cute import size, rank, make_identity_tensor


@cute.jit
def _extract_layout_indices(layout, M, N):
    """
    Extract indices from a layout using compile-time loops.

    Args:
        layout: CuTe layout object
        M: Number of rows (must be compile-time constant)
        N: Number of columns (must be compile-time constant)

    Returns:
        2D numpy array of indices
    """
    indices = np.zeros((M, N), dtype=np.int32)

    for i in range_constexpr(M):
        for j in range_constexpr(N):
            indices[i, j] = layout((i, j))

    return indices


def _create_layout_svg(layout):
    """
    Internal helper to create SVG Drawing object for a layout.

    Args:
        layout: CuTe layout object

    Returns:
        svgwrite.Drawing object
    """
    # 8 RGB-255 Greyscale colors
    rgb_255_colors = [
        (255, 255, 255),
        (230, 230, 230),
        (205, 205, 205),
        (180, 180, 180),
        (155, 155, 155),
        (130, 130, 130),
        (105, 105, 105),
        (80, 80, 80),
    ]

    cell_size = 20
    M, N = size(layout[0]), size(layout[1])

    # Extract indices using JIT-compiled function with range_constexpr
    indices = _extract_layout_indices(layout, M, N)

    dwg = svgwrite.Drawing(size=(N * cell_size, M * cell_size))

    for i in range(M):
        for j in range(N):
            idx = indices[i, j]
            x = j * cell_size
            y = i * cell_size

            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(
                        *rgb_255_colors[idx % len(rgb_255_colors)], mode="RGB"
                    ),
                    stroke="black",
                )
            )

            dwg.add(
                dwg.text(
                    str(idx),
                    insert=(x + cell_size // 2, y + cell_size // 2),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    return dwg


@cute.jit
def _extract_tv_layout_coords(layout, num_threads, num_values):
    """
    Extract thread-value layout coordinates using compile-time loops.

    Args:
        layout: CuTe TV layout object
        num_threads: Number of threads
        num_values: Number of values per thread

    Returns:
        2D numpy array of (i, j) coordinates for each (tid, vid) pair
    """
    coords = np.zeros((num_threads, num_values, 2), dtype=np.int32)

    for tid in range_constexpr(num_threads):
        for vid in range_constexpr(num_values):
            i, j = layout[(tid, vid)]
            coords[tid, vid, 0] = i
            coords[tid, vid, 1] = j

    return coords


@cute.jit
def _extract_copy_layout_coords(layout_s, layout_d, num_threads, num_values):
    """
    Extract coordinates from source and destination TV layouts using compile-time loops.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        num_threads: Number of threads
        num_values: Number of values per thread

    Returns:
        Tuple of two 2D numpy arrays of (i, j) coordinates for each (tid, vid) pair
    """
    coords_s = np.zeros((num_threads, num_values, 2), dtype=np.int32)
    coords_d = np.zeros((num_threads, num_values, 2), dtype=np.int32)

    for tid in range_constexpr(num_threads):
        for vid in range_constexpr(num_values):
            i_s, j_s = layout_s[(tid, vid)]
            coords_s[tid, vid, 0] = i_s
            coords_s[tid, vid, 1] = j_s

            i_d, j_d = layout_d[(tid, vid)]
            coords_d[tid, vid, 0] = i_d
            coords_d[tid, vid, 1] = j_d

    return coords_s, coords_d


def _create_tv_layout_svg(layout, tile_mn):
    """
    Internal helper to create SVG Drawing object for a TV layout.

    Args:
        layout: CuTe layout object (rank-2 TV Layout)
        tile_mn: Rank-2 MN Tile

    Returns:
        svgwrite.Drawing object
    """
    assert rank(layout) == 2, "Expected a rank-2 TV Layout"
    assert rank(tile_mn) == 2, "Expected a rank-2 MN Tile"

    coord = make_identity_tensor(tile_mn)
    layout = cute.composition(coord, layout)

    # 8 RGB-255 colors
    rgb_255_colors = [
        (175, 175, 255),
        (175, 255, 175),
        (255, 255, 175),
        (255, 175, 175),
        (210, 210, 255),
        (210, 255, 210),
        (255, 255, 210),
        (255, 210, 210),
    ]

    cell_size = 20
    M, N = size(tile_mn[0]), size(tile_mn[1])
    num_threads = size(layout, mode=[0])
    num_values = size(layout, mode=[1])

    # Extract coordinates using JIT-compiled function with range_constexpr
    coords = _extract_tv_layout_coords(layout, num_threads, num_values)

    filled = np.zeros((M, N), dtype=bool)
    dwg = svgwrite.Drawing(size=(N * cell_size, M * cell_size))

    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(j * cell_size, i * cell_size),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords[tid, vid, 0]), int(coords[tid, vid, 1])
            x = j * cell_size
            y = i * cell_size

            if filled[i, j]:
                continue
            filled[i, j] = True

            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(
                        *rgb_255_colors[tid % len(rgb_255_colors)], mode="RGB"
                    ),
                    stroke="black",
                )
            )

            dwg.add(
                dwg.text(
                    f"T{tid}",
                    insert=(x + cell_size // 2, y + 1 * cell_size // 4),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )
            dwg.add(
                dwg.text(
                    f"V{vid}",
                    insert=(x + cell_size // 2, y + 3 * cell_size // 4),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    return dwg


def render_layout_svg(layout, output_file):
    """
    Render a CuTe layout as an SVG grid with color-coded cells.

    Args:
        layout: CuTe layout object
        output_file: Output SVG file path
    """
    dwg = _create_layout_svg(layout)
    dwg.saveas(output_file)


def render_tv_layout_svg(layout, tile_mn, output_file):
    """
    Render a CuTe thread-value (TV) layout as an SVG grid.

    Args:
        layout: CuTe layout object (rank-2 TV Layout)
        tile_mn: Rank-2 MN Tile
        output_file: Output SVG file path
    """
    dwg = _create_tv_layout_svg(layout, tile_mn)
    dwg.saveas(output_file)


def display_svg(file_path):
    """
    Display an SVG file in Jupyter notebooks.

    Args:
        file_path: Path to the SVG file to display

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    with open(file_path, "r") as f:
        svg_content = f.read()

    return display(SVG(svg_content))


def display_layout(layout):
    """
    Display a CuTe layout directly in Jupyter notebooks without writing to disk.

    Args:
        layout: CuTe layout object

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    dwg = _create_layout_svg(layout)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


def display_tv_layout(layout, tile_mn):
    """
    Display a CuTe thread-value (TV) layout directly in Jupyter notebooks without writing to disk.

    Args:
        layout: CuTe layout object (rank-2 TV Layout)
        tile_mn: Rank-2 MN Tile

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    dwg = _create_tv_layout_svg(layout, tile_mn)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


@cute.jit
def _extract_swizzle_indices(layout, M, N):
    """
    Extract indices from a swizzle layout using compile-time loops.

    This function must be JIT-compiled and use range_constexpr to ensure
    the swizzle transformation is properly evaluated at compile time.

    Args:
        layout: CuTe swizzle/composed layout object
        M: Number of rows (must be compile-time constant)
        N: Number of columns (must be compile-time constant)

    Returns:
        2D numpy array of indices
    """
    # Create output array
    indices = np.zeros((M, N), dtype=np.int32)

    # Use range_constexpr for compile-time unrolling
    # This ensures swizzle is evaluated at compile time
    for i in range_constexpr(M):
        for j in range_constexpr(N):
            indices[i, j] = layout((i, j))

    return indices


def _create_swizzle_layout_svg(layout):
    """
    Internal helper to create SVG Drawing object for a Swizzle layout.

    Swizzle layouts are special composed layouts that permute elements
    to improve memory access patterns. This function uses compile-time
    loop unrolling to properly evaluate the swizzle transformation.

    Args:
        layout: CuTe swizzle layout object

    Returns:
        svgwrite.Drawing object
    """
    # 8 RGB-255 Greyscale colors (same as basic layout)
    rgb_255_colors = [
        (255, 255, 255),
        (230, 230, 230),
        (205, 205, 205),
        (180, 180, 180),
        (155, 155, 155),
        (130, 130, 130),
        (105, 105, 105),
        (80, 80, 80),
    ]

    cell_size = 20
    M, N = size(layout[0]), size(layout[1])

    # Extract indices using JIT-compiled function with range_constexpr
    indices = _extract_swizzle_indices(layout, M, N)

    dwg = svgwrite.Drawing(size=(N * cell_size, M * cell_size))

    for i in range(M):
        for j in range(N):
            idx = indices[i, j]
            x = j * cell_size
            y = i * cell_size

            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(
                        *rgb_255_colors[idx % len(rgb_255_colors)], mode="RGB"
                    ),
                    stroke="black",
                )
            )

            dwg.add(
                dwg.text(
                    str(idx),
                    insert=(x + cell_size // 2, y + cell_size // 2),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    return dwg


def render_swizzle_layout_svg(layout, output_file):
    """
    Render a CuTe Swizzle layout as an SVG grid with color-coded cells.

    Swizzle layouts permute elements to improve memory access patterns.
    This function visualizes the swizzled memory layout pattern.

    Args:
        layout: CuTe swizzle layout object
        output_file: Output SVG file path
    """
    dwg = _create_swizzle_layout_svg(layout)
    dwg.saveas(output_file)


def display_swizzle_layout(layout):
    """
    Display a CuTe Swizzle layout directly in Jupyter notebooks without writing to disk.

    Args:
        layout: CuTe swizzle layout object

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    dwg = _create_swizzle_layout_svg(layout)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


def _create_copy_layout_svg(layout_s, layout_d, tile_mn):
    """
    Internal helper to create SVG Drawing object for a Copy layout.

    Copy layouts show source and destination thread-value mappings side-by-side,
    visualizing how data is copied from source to destination memory locations.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        tile_mn: Rank-2 MN Tile

    Returns:
        svgwrite.Drawing object
    """
    assert rank(layout_s) == 2, "Expected a rank-2 source TV Layout"
    assert rank(layout_d) == 2, "Expected a rank-2 destination TV Layout"
    assert rank(tile_mn) == 2, "Expected a rank-2 MN Tile"

    # 8 RGB-255 colors (same as TV layout)
    rgb_255_colors = [
        (175, 175, 255),
        (175, 255, 175),
        (255, 255, 175),
        (255, 175, 175),
        (210, 210, 255),
        (210, 255, 210),
        (255, 255, 210),
        (255, 210, 210),
    ]

    cell_size = 20
    M, N = size(tile_mn[0]), size(tile_mn[1])
    num_threads = size(layout_s, mode=[0])
    num_values = size(layout_s, mode=[1])

    # Extract coordinates using JIT-compiled function with range_constexpr
    coords_s, coords_d = _extract_copy_layout_coords(
        layout_s, layout_d, num_threads, num_values
    )

    # Horizontal gap between source and destination grids
    gap = 3 * cell_size
    total_width = 2 * N * cell_size + gap

    filled_s = np.zeros((M, N), dtype=bool)
    filled_d = np.zeros((M, N), dtype=bool)
    dwg = svgwrite.Drawing(size=(total_width, M * cell_size))

    # Draw source grid (left side) - background cells
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(j * cell_size, i * cell_size),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Draw destination grid (right side) - background cells
    x_offset = N * cell_size + gap
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(x_offset + j * cell_size, i * cell_size),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Fill source grid with thread-value data
    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords_s[tid, vid, 0]), int(coords_s[tid, vid, 1])
            x = j * cell_size
            y = i * cell_size

            if filled_s[i, j]:
                continue
            filled_s[i, j] = True

            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(
                        *rgb_255_colors[tid % len(rgb_255_colors)], mode="RGB"
                    ),
                    stroke="black",
                )
            )

            dwg.add(
                dwg.text(
                    f"T{tid}",
                    insert=(x + cell_size // 2, y + 1 * cell_size // 4),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )
            dwg.add(
                dwg.text(
                    f"V{vid}",
                    insert=(x + cell_size // 2, y + 3 * cell_size // 4),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    # Fill destination grid with thread-value data
    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords_d[tid, vid, 0]), int(coords_d[tid, vid, 1])
            x = x_offset + j * cell_size
            y = i * cell_size

            if filled_d[i, j]:
                continue
            filled_d[i, j] = True

            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(
                        *rgb_255_colors[tid % len(rgb_255_colors)], mode="RGB"
                    ),
                    stroke="black",
                )
            )

            dwg.add(
                dwg.text(
                    f"T{tid}",
                    insert=(x + cell_size // 2, y + 1 * cell_size // 4),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )
            dwg.add(
                dwg.text(
                    f"V{vid}",
                    insert=(x + cell_size // 2, y + 3 * cell_size // 4),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    return dwg


def render_copy_layout_svg(layout_s, layout_d, tile_mn, output_file):
    """
    Render a CuTe copy layout as an SVG with source and destination grids side-by-side.

    Copy layouts show how threads map to source and destination memory locations,
    visualizing the data movement pattern.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        tile_mn: Rank-2 MN Tile
        output_file: Output SVG file path
    """
    dwg = _create_copy_layout_svg(layout_s, layout_d, tile_mn)
    dwg.saveas(output_file)


def display_copy_layout(layout_s, layout_d, tile_mn):
    """
    Display a CuTe copy layout directly in Jupyter notebooks without writing to disk.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        tile_mn: Rank-2 MN Tile

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    dwg = _create_copy_layout_svg(layout_s, layout_d, tile_mn)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


###################################
# TiledCopy Layout Extraction Utilities
###################################


def tidfrg_S(tiled_copy, tile_mn):
    """
    Extract source thread-value layout from TiledCopy.

    Python equivalent of C++ TiledCopy::tidfrg_S().

    This function creates a tensor that maps (thread_id, value_id) coordinates
    to (m, n) spatial coordinates for the source layout of a copy operation.

    Args:
        tiled_copy: CuTe TiledCopy object
        tile_mn: Tile shape as (M, N) tuple or Shape object

    Returns:
        Tensor mapping (thread_id, value_id) -> (m, n) coordinates for source

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import tidfrg_S
        >>>
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> layout_s = tidfrg_S(tiled_copy, (8, 8))
    """
    ref_tensor = make_identity_tensor(tile_mn)
    return cute.composition(ref_tensor, tiled_copy.layout_src_tv_tiled)


def tidfrg_D(tiled_copy, tile_mn):
    """
    Extract destination thread-value layout from TiledCopy.

    Python equivalent of C++ TiledCopy::tidfrg_D().

    This function creates a tensor that maps (thread_id, value_id) coordinates
    to (m, n) spatial coordinates for the destination layout of a copy operation.

    Args:
        tiled_copy: CuTe TiledCopy object
        tile_mn: Tile shape as (M, N) tuple or Shape object

    Returns:
        Tensor mapping (thread_id, value_id) -> (m, n) coordinates for destination

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import tidfrg_D
        >>>
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> layout_d = tidfrg_D(tiled_copy, (8, 8))
    """
    ref_tensor = make_identity_tensor(tile_mn)
    return cute.composition(ref_tensor, tiled_copy.layout_dst_tv_tiled)


###################################
# High-Level TiledCopy Visualization API
###################################


def render_tiled_copy_svg(tiled_copy, tile_mn, output_path):
    """
    Render a TiledCopy visualization to SVG file.

    Python equivalent of C++ print_latex(TiledCopy).

    This high-level function automatically extracts source and destination
    thread-value layouts from a TiledCopy object and renders them side-by-side.

    Args:
        tiled_copy: CuTe TiledCopy object created with make_tiled_copy_tv()
        tile_mn: Tile shape as (M, N) tuple
        output_path: Path to save the SVG file

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import render_tiled_copy_svg
        >>>
        >>> # Create copy atom and tiled copy
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> # Render in one call!
        >>> render_tiled_copy_svg(tiled_copy, (8, 8), "copy_layout.svg")
    """
    # Extract source and destination TV layouts
    tensor_s_tv = tidfrg_S(tiled_copy, tile_mn)
    tensor_d_tv = tidfrg_D(tiled_copy, tile_mn)

    # Handle potential extra dimensions (like C++ (_,_,Int<0>{}))
    # If tensors have more than 2 dimensions, slice to get the first element
    if hasattr(tensor_s_tv, 'ndim') and tensor_s_tv.ndim > 2:
        tensor_s_tv = tensor_s_tv[:, :, 0]
    if hasattr(tensor_d_tv, 'ndim') and tensor_d_tv.ndim > 2:
        tensor_d_tv = tensor_d_tv[:, :, 0]

    # Render the copy layout
    render_copy_layout_svg(tensor_s_tv, tensor_d_tv, tile_mn, output_path)


def display_tiled_copy(tiled_copy, tile_mn):
    """
    Display a TiledCopy visualization in Jupyter notebook.

    Python equivalent of C++ print_latex(TiledCopy) for interactive notebooks.

    Args:
        tiled_copy: CuTe TiledCopy object created with make_tiled_copy_tv()
        tile_mn: Tile shape as (M, N) tuple

    Returns:
        IPython.display.SVG object for inline display

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import display_tiled_copy
        >>>
        >>> # Create copy atom and tiled copy
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> # Display inline in Jupyter
        >>> display_tiled_copy(tiled_copy, (8, 8))
    """
    # Extract source and destination TV layouts
    tensor_s_tv = tidfrg_S(tiled_copy, tile_mn)
    tensor_d_tv = tidfrg_D(tiled_copy, tile_mn)

    # Handle potential extra dimensions
    if hasattr(tensor_s_tv, 'ndim') and tensor_s_tv.ndim > 2:
        tensor_s_tv = tensor_s_tv[:, :, 0]
    if hasattr(tensor_d_tv, 'ndim') and tensor_d_tv.ndim > 2:
        tensor_d_tv = tensor_d_tv[:, :, 0]

    # Display the copy layout
    return display_copy_layout(tensor_s_tv, tensor_d_tv, tile_mn)


###################################
# MMA (Matrix Multiply-Accumulate) Visualization
###################################


@cute.jit
def _extract_mma_coords(tensorC, tensorA, tensorB, num_threads_C, num_values_C, num_threads_A, num_values_A, num_threads_B, num_values_B):
    """
    Extract coordinates from A, B, and C tensors using compile-time loops.

    Args:
        tensorC: C matrix TV tensor
        tensorA: A matrix TV tensor
        tensorB: B matrix TV tensor
        num_threads_C, num_values_C: Thread and value counts for C
        num_threads_A, num_values_A: Thread and value counts for A
        num_threads_B, num_values_B: Thread and value counts for B

    Returns:
        Tuple of three arrays with coordinates for C, A, B
    """
    coords_C = np.zeros((num_threads_C, num_values_C, 2), dtype=np.int32)
    coords_A = np.zeros((num_threads_A, num_values_A, 2), dtype=np.int32)
    coords_B = np.zeros((num_threads_B, num_values_B, 2), dtype=np.int32)

    for tid in range_constexpr(num_threads_C):
        for vid in range_constexpr(num_values_C):
            m, n = tensorC[tid, vid]
            coords_C[tid, vid, 0] = m
            coords_C[tid, vid, 1] = n

    for tid in range_constexpr(num_threads_A):
        for vid in range_constexpr(num_values_A):
            m, k = tensorA[tid, vid]
            coords_A[tid, vid, 0] = m
            coords_A[tid, vid, 1] = k

    for tid in range_constexpr(num_threads_B):
        for vid in range_constexpr(num_values_B):
            n, k = tensorB[tid, vid]
            coords_B[tid, vid, 0] = n
            coords_B[tid, vid, 1] = k

    return coords_C, coords_A, coords_B


def _create_mma_layout_svg(tiled_mma, tile_mnk):
    """
    Create SVG visualization of MMA layout showing A, B, and C matrices.

    Layout:
        B
    A   C

    Where C = A × B for matrix multiplication.

    Args:
        tiled_mma: TiledMMA object
        tile_mnk: Tuple of (M, N, K) tile dimensions

    Returns:
        svgwrite.Drawing object
    """
    M, N, K = tile_mnk

    # Extract TV layouts from TiledMMA
    layoutC_TV = tiled_mma.tv_layout_C_tiled
    layoutA_TV = tiled_mma.tv_layout_A_tiled
    layoutB_TV = tiled_mma.tv_layout_B_tiled

    # Create identity tensors and compose
    refC = make_identity_tensor((M, N))
    tensorC_TV = cute.composition(refC, layoutC_TV)

    refA = make_identity_tensor((M, K))
    tensorA_TV = cute.composition(refA, layoutA_TV)

    refB = make_identity_tensor((N, K))
    tensorB_TV = cute.composition(refB, layoutB_TV)

    # Handle potential extra dimensions
    tensorC = tensorC_TV[:, :, 0] if hasattr(tensorC_TV, 'ndim') and tensorC_TV.ndim > 2 else tensorC_TV
    tensorA = tensorA_TV[:, :, 0] if hasattr(tensorA_TV, 'ndim') and tensorA_TV.ndim > 2 else tensorA_TV
    tensorB = tensorB_TV[:, :, 0] if hasattr(tensorB_TV, 'ndim') and tensorB_TV.ndim > 2 else tensorB_TV

    cell_size = 20

    # SVG dimensions
    page_width = (K + N + 2) * cell_size
    page_height = (K + M + 2) * cell_size

    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Track filled cells to avoid duplicates
    import numpy as np
    filled = np.zeros((M, N, K), dtype=bool)

    # Get number of threads and values for each tensor
    num_threads_C = size(tensorC, mode=[0])
    num_values_C = size(tensorC, mode=[1])
    num_threads_A = size(tensorA, mode=[0])
    num_values_A = size(tensorA, mode=[1])
    num_threads_B = size(tensorB, mode=[0])
    num_values_B = size(tensorB, mode=[1])

    # Extract coordinates from tensors (this happens in JIT context)
    coords_C, coords_A, coords_B = _extract_mma_coords(
        tensorC, tensorA, tensorB,
        num_threads_C, num_values_C,
        num_threads_A, num_values_A,
        num_threads_B, num_values_B
    )

    # 8 RGB-255 pastel colors (matching TV layout)
    rgb_255_colors = [
        (175, 175, 255),
        (175, 255, 175),
        (255, 255, 175),
        (255, 175, 175),
        (255, 175, 255),
        (175, 255, 255),
        (210, 210, 210),
        (160, 160, 255),
    ]

    # --- Draw C (M×N at bottom-right) ---
    for tid in range(num_threads_C):
        for vid in range(num_values_C):
            m, n = int(coords_C[tid, vid, 0]), int(coords_C[tid, vid, 1])
            if m < M and n < N and not filled[m, n, 0]:
                filled[m, n, 0] = True

                x = (n + K + 2) * cell_size
                y = (m + K + 2) * cell_size

                color = rgb_255_colors[tid % len(rgb_255_colors)]

                rect = dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(*color, mode="RGB"),
                    stroke='black'
                )
                dwg.add(rect)

                # Thread ID
                text1 = dwg.text(
                    f'T{tid}',
                    insert=(x + cell_size/2, y + cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text1)

                # Value ID
                text2 = dwg.text(
                    f'V{vid}',
                    insert=(x + cell_size/2, y + 3*cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text2)

    # Reset filled tracker
    filled.fill(False)

    # --- Draw A (M×K at left) ---
    for tid in range(num_threads_A):
        for vid in range(num_values_A):
            m, k = int(coords_A[tid, vid, 0]), int(coords_A[tid, vid, 1])
            if m < M and k < K and not filled[m, 0, k]:
                filled[m, 0, k] = True

                x = (k + 1) * cell_size
                y = (m + K + 2) * cell_size

                color = rgb_255_colors[tid % len(rgb_255_colors)]

                rect = dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(*color, mode="RGB"),
                    stroke='black'
                )
                dwg.add(rect)

                # Thread ID
                text1 = dwg.text(
                    f'T{tid}',
                    insert=(x + cell_size/2, y + cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text1)

                # Value ID
                text2 = dwg.text(
                    f'V{vid}',
                    insert=(x + cell_size/2, y + 3*cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text2)

    # Reset filled tracker
    filled.fill(False)

    # --- Draw B (N×K at top, shown as K×N transposed) ---
    for tid in range(num_threads_B):
        for vid in range(num_values_B):
            n, k = int(coords_B[tid, vid, 0]), int(coords_B[tid, vid, 1])
            if n < N and k < K and not filled[0, n, k]:
                filled[0, n, k] = True

                x = (n + K + 2) * cell_size
                y = (k + 1) * cell_size

                color = rgb_255_colors[tid % len(rgb_255_colors)]

                rect = dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(*color, mode="RGB"),
                    stroke='black'
                )
                dwg.add(rect)

                # Thread ID
                text1 = dwg.text(
                    f'T{tid}',
                    insert=(x + cell_size/2, y + cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text1)

                # Value ID
                text2 = dwg.text(
                    f'V{vid}',
                    insert=(x + cell_size/2, y + 3*cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text2)

    return dwg


def render_mma_layout_svg(tiled_mma, tile_mnk, output_file):
    """
    Render a TiledMMA layout as an SVG showing A, B, and C matrix thread mappings.

    Alias for render_tiled_mma_svg().

    Args:
        tiled_mma: CuTe TiledMMA object
        tile_mnk: Tuple (M, N, K) tile dimensions
        output_file: Output SVG file path
    """
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    dwg.saveas(output_file)


def display_mma_layout(tiled_mma, tile_mnk):
    """
    Display a TiledMMA layout directly in Jupyter notebooks.

    Alias for display_tiled_mma().

    Args:
        tiled_mma: CuTe TiledMMA object
        tile_mnk: Tuple (M, N, K) tile dimensions

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    return SVG(dwg.tostring())


###################################
# High-Level TiledMMA Visualization API
###################################


def render_tiled_mma_svg(tiled_mma, tile_mnk, output_path):
    """
    Render a TiledMMA visualization to SVG file.

    Python equivalent of C++ print_latex(TiledMMA).

    This high-level function automatically extracts A, B, and C thread-value
    layouts from a TiledMMA object and renders them in the standard layout:
        B
    A   C

    Args:
        tiled_mma: CuTe TiledMMA object created with make_tiled_mma()
        tile_mnk: Tile shape as (M, N, K) tuple
        output_path: Path to save the SVG file

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import render_tiled_mma_svg
        >>>
        >>> # Create MMA operation and tiled MMA
        >>> op = cute.nvgpu.MmaUniversalOp(Float32)
        >>> atoms_layout = cute.make_layout((16, 1, 1), stride=(1, 0, 0))
        >>> tiled_mma = cute.make_tiled_mma(op, atoms_layout)
        >>>
        >>> # Render in one call!
        >>> render_tiled_mma_svg(tiled_mma, (8, 8, 8), "mma_layout.svg")
    """
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    dwg.saveas(output_path)


def display_tiled_mma(tiled_mma, tile_mnk):
    """
    Display a TiledMMA visualization in Jupyter notebook.

    Python equivalent of C++ print_latex(TiledMMA) for interactive notebooks.

    Args:
        tiled_mma: CuTe TiledMMA object created with make_tiled_mma()
        tile_mnk: Tile shape as (M, N, K) tuple

    Returns:
        IPython.display.SVG object for inline display

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import display_tiled_mma
        >>>
        >>> # Create MMA operation and tiled MMA
        >>> op = cute.nvgpu.MmaUniversalOp(Float32)
        >>> atoms_layout = cute.make_layout((16, 1, 1), stride=(1, 0, 0))
        >>> tiled_mma = cute.make_tiled_mma(op, atoms_layout)
        >>>
        >>> # Display inline in Jupyter
        >>> display_tiled_mma(tiled_mma, (8, 8, 8))
    """
    from IPython.display import SVG
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    return SVG(dwg.tostring())
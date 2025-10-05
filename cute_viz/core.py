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
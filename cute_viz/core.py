"""
Core visualization functions for CuTe layouts.
"""

import numpy as np
import svgwrite
from cutlass import cute
from cutlass.cute import size, rank, make_identity_tensor


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
    dwg = svgwrite.Drawing(size=(N * cell_size, M * cell_size))

    for i in range(M):
        for j in range(N):
            idx = layout((i, j))
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

    for tid in range(size(layout, mode=[0])):
        for vid in range(size(layout, mode=[1])):
            i, j = layout[(tid, vid)]
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
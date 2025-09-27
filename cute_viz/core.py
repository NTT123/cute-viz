"""
Core visualization functions for CuTe layouts.
"""

import numpy as np
import svgwrite
from cutlass import cute
from cutlass.cute import size, rank, make_identity_tensor


def render_layout_svg(layout, output_file):
    """
    Render a CuTe layout as an SVG grid with color-coded cells.

    Args:
        layout: CuTe layout object
        output_file: Output SVG file path
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

    # Cell size in pixels
    cell_size = 20

    # Grid size
    M, N = size(layout[0]), size(layout[1])

    # Create SVG canvas
    dwg = svgwrite.Drawing(output_file, size=(N * cell_size, M * cell_size))

    # Draw grid cells
    for i in range(M):
        for j in range(N):
            idx = layout((i, j))
            x = j * cell_size
            y = i * cell_size

            # Draw rectangle
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

            # Add label text
            dwg.add(
                dwg.text(
                    str(idx),
                    insert=(x + cell_size // 2, y + cell_size // 2),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    dwg.save()


def render_tv_layout_svg(layout, tile_mn, output_file):
    """
    Render a CuTe thread-value (TV) layout as an SVG grid.

    Args:
        layout: CuTe layout object (rank-2 TV Layout)
        tile_mn: Rank-2 MN Tile
        output_file: Output SVG file path
    """
    assert rank(layout) == 2, "Expected a rank-2 TV Layout"
    assert rank(tile_mn) == 2, "Expected a rank-2 MN Tile"

    coord = make_identity_tensor(tile_mn)
    layout = cute.composition(coord, layout)
    # assert congruent(coprofile(layout), (0,0)), "Expected a 2D codomain (tid,vid) -> (m,n)"

    # 8 RGB-255 colors, TODO Generalize
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

    # Cell size in pixels
    cell_size = 20

    # Grid size
    M, N = size(tile_mn[0]), size(tile_mn[1])
    filled = np.zeros((M, N), dtype=bool)

    # Create SVG canvas
    dwg = svgwrite.Drawing(output_file, size=(N * cell_size, M * cell_size))

    # Fill in grid
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

    # Draw TV cells
    for tid in range(size(layout, mode=[0])):
        for vid in range(size(layout, mode=[1])):
            i, j = layout[(tid, vid)]
            x = j * cell_size
            y = i * cell_size

            if filled[i, j]:
                continue
            filled[i, j] = True

            # Draw rectangle
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

            # Add label text
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

    dwg.save()


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

    # Cell size in pixels
    cell_size = 20

    # Grid size
    M, N = size(layout[0]), size(layout[1])

    # Create SVG canvas
    dwg = svgwrite.Drawing(size=(N * cell_size, M * cell_size))

    # Draw grid cells
    for i in range(M):
        for j in range(N):
            idx = layout((i, j))
            x = j * cell_size
            y = i * cell_size

            # Draw rectangle
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

            # Add label text
            dwg.add(
                dwg.text(
                    str(idx),
                    insert=(x + cell_size // 2, y + cell_size // 2),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    # Get SVG as string and display directly
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

    assert rank(layout) == 2, "Expected a rank-2 TV Layout"
    assert rank(tile_mn) == 2, "Expected a rank-2 MN Tile"

    coord = make_identity_tensor(tile_mn)
    layout = cute.composition(coord, layout)
    # assert congruent(coprofile(layout), (0,0)), "Expected a 2D codomain (tid,vid) -> (m,n)"

    # 8 RGB-255 colors, TODO Generalize
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

    # Cell size in pixels
    cell_size = 20

    # Grid size
    M, N = size(tile_mn[0]), size(tile_mn[1])
    filled = np.zeros((M, N), dtype=bool)

    # Create SVG canvas
    dwg = svgwrite.Drawing(size=(N * cell_size, M * cell_size))

    # Fill in grid
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

    # Draw TV cells
    for tid in range(size(layout, mode=[0])):
        for vid in range(size(layout, mode=[1])):
            i, j = layout[(tid, vid)]
            x = j * cell_size
            y = i * cell_size

            if filled[i, j]:
                continue
            filled[i, j] = True

            # Draw rectangle
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

            # Add label text
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

    # Get SVG as string and display directly
    svg_string = dwg.tostring()
    return display(SVG(svg_string))
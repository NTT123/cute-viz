"""
Core visualization functions for CuTe layouts.
"""

import numpy as np
import svgwrite
from cutlass import cute, range_constexpr
from cutlass.cute import size, rank, make_identity_tensor, idx2crd, depth


@cute.jit
def _extract_layout_indices_universal(layout, total_size):
    """
    Universal function to extract indices from any layout using idx2crd.
    
    This works for layouts of any rank (1D, 2D, 3D, 4D, ...) and 
    any nesting structure (hierarchical layouts like (2, (2, 2))).
    
    The approach:
    1. Iterate through all linear indices 0 to total_size-1
    2. Use idx2crd to convert each linear index to a coordinate using layout.shape
    3. Apply the layout to that coordinate to get the output index

    Args:
        layout: CuTe layout object (any rank, any nesting)
        total_size: Total number of elements (must be compile-time constant)

    Returns:
        1D numpy array of indices
    """
    indices = np.zeros(total_size, dtype=np.int32)

    for i in range_constexpr(total_size):
        # Convert linear index to coordinate using the layout's shape
        coord = idx2crd(i, layout.shape)
        # Apply layout to coordinate to get output index
        indices[i] = layout(coord)

    return indices


def _extract_layout_indices(layout):
    """
    Universal function to extract indices from a layout of any rank and structure.
    
    Works for:
    - Simple 1D layouts: 8:1
    - Simple 2D layouts: (4,8):(1,4)
    - Simple 3D+ layouts: (2,4,8):(1,8,32)
    - Hierarchical/nested layouts: (2,(2,2)):(4,(2,1))
    - Any combination of the above

    Args:
        layout: CuTe layout object (any rank, any nesting)

    Returns:
        1D numpy array of indices (flattened)
    """
    # Get total size of the layout and convert to Python int
    total_size = int(size(layout))
    
    # Extract all indices using universal extraction
    return _extract_layout_indices_universal(layout, total_size)


def _create_layout_svg(layout, flatten_hierarchical=True):
    """
    Universal function to create SVG Drawing for any CuTe layout.
    
    Handles layouts of any rank and structure:
    - Rank 1: horizontal bar with top labels
    - Rank 2: 2D grid with top and left labels (handles nested shapes)
    - Rank 3+: horizontal slices showing first dimension as slices

    Args:
        layout: CuTe layout object (any rank, any nesting)
        flatten_hierarchical: If True (default), hierarchical layouts are rendered as flat grids.
                             If False, hierarchical layouts are rendered with tile boundaries.

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
    layout_rank = int(rank(layout))
    
    # Extract all indices (flattened) using universal extraction
    indices_flat = _extract_layout_indices(layout)
    
    # Dispatch based on rank
    if layout_rank == 1:
        # 1D layout: horizontal bar
        N = int(size(layout))
        indices = indices_flat  # Already 1D
        
        label_margin = cell_size
        page_width = N * cell_size
        page_height = cell_size + label_margin
        
        dwg = svgwrite.Drawing(size=(page_width, page_height))
        
        # Draw cells
        for i in range(N):
            idx = indices[i]
            x = i * cell_size
            y = label_margin
            
            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(*rgb_255_colors[idx % len(rgb_255_colors)], mode="RGB"),
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
        
        # Top labels
        for i in range(N):
            x = i * cell_size + cell_size // 2
            y = label_margin // 2
            dwg.add(
                dwg.text(
                    str(i),
                    insert=(x, y),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )
        
        return dwg
    
    elif layout_rank == 2:
        # Check if this is a hierarchical 2D layout and user wants nested visualization
        if not flatten_hierarchical and _is_hierarchical_2d(layout):
            # Use special tiled visualization for hierarchical layouts
            dwg = _create_hierarchical_2d_layout_svg(layout)
            if dwg is not None:
                return dwg
        
        # Regular 2D layout: grid (or flattened hierarchical)
        M, N = int(size(layout[0])), int(size(layout[1]))
        # Reshape using column-major (Fortran) order since idx2crd uses column-major
        indices = indices_flat.reshape(M, N, order='F')
        
        label_margin = cell_size
        page_width = N * cell_size + label_margin
        page_height = M * cell_size + label_margin
        
        dwg = svgwrite.Drawing(size=(page_width, page_height))
        
        # Draw grid cells
        for i in range(M):
            for j in range(N):
                idx = indices[i, j]
                x = j * cell_size + label_margin
                y = i * cell_size + label_margin
                
                dwg.add(
                    dwg.rect(
                        insert=(x, y),
                        size=(cell_size, cell_size),
                        fill=svgwrite.rgb(*rgb_255_colors[idx % len(rgb_255_colors)], mode="RGB"),
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
        
        # Top labels
        for j in range(N):
            x = j * cell_size + label_margin + cell_size // 2
            y = label_margin // 2
            dwg.add(
                dwg.text(
                    str(j),
                    insert=(x, y),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )
        
        # Left labels
        for i in range(M):
            x = label_margin // 2
            y = i * cell_size + label_margin + cell_size // 2
            dwg.add(
                dwg.text(
                    str(i),
                    insert=(x, y),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )
        
        return dwg
    
    else:
        # 3D+ layout: visualize as slices
        shape_sizes = _get_layout_shape_sizes(layout)
        
        # First dimension = number of slices
        D0 = shape_sizes[0]
        
        # Flatten remaining dimensions to 2D
        D1 = shape_sizes[1]
        D2 = 1
        for i in range(2, len(shape_sizes)):
            D2 *= shape_sizes[i]
        
        if len(shape_sizes) == 3:
            D2 = shape_sizes[2]
        
        # Reshape indices
        total_size = int(size(layout))
        if total_size != D0 * D1 * D2:
            D2 = total_size // (D0 * D1)
        
        # Reshape using column-major (Fortran) order since idx2crd uses column-major
        indices = indices_flat.reshape(D0, D1, D2, order='F')
        
        # Layout slices horizontally
        slice_spacing = cell_size
        label_margin = cell_size
        
        slice_width = D2 * cell_size + label_margin
        slice_height = D1 * cell_size + label_margin
        
        page_width = D0 * slice_width + (D0 - 1) * slice_spacing
        page_height = slice_height
        
        dwg = svgwrite.Drawing(size=(page_width, page_height))
        
        # Draw each slice
        for d in range(D0):
            slice_offset_x = d * (slice_width + slice_spacing)
            
            for i in range(D1):
                for j in range(D2):
                    idx = indices[d, i, j]
                    x = slice_offset_x + j * cell_size + label_margin
                    y = i * cell_size + label_margin
                    
                    dwg.add(
                        dwg.rect(
                            insert=(x, y),
                            size=(cell_size, cell_size),
                            fill=svgwrite.rgb(*rgb_255_colors[idx % len(rgb_255_colors)], mode="RGB"),
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
            
            # Top labels
            for j in range(D2):
                x = slice_offset_x + j * cell_size + label_margin + cell_size // 2
                y = label_margin // 2
                dwg.add(
                    dwg.text(
                        str(j),
                        insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px",
                    )
                )
            
            # Left labels
            for i in range(D1):
                x = slice_offset_x + label_margin // 2
                y = i * cell_size + label_margin + cell_size // 2
                dwg.add(
                    dwg.text(
                        str(i),
                        insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px",
                    )
                )
            
            # Slice label
            slice_label_x = slice_offset_x + slice_width // 2
            slice_label_y = page_height - 5
            dwg.add(
                dwg.text(
                    f"[{d},:,:]",
                    insert=(slice_label_x, slice_label_y),
                    text_anchor="middle",
                    font_size="10px",
                    font_weight="bold",
                )
            )
        
        return dwg


def _get_layout_shape_sizes(layout):
    """
    Get the sizes of each mode in a layout, handling any rank.
    
    Args:
        layout: CuTe layout object
    
    Returns:
        List of sizes for each mode (as Python ints)
    """
    layout_rank = int(rank(layout))
    if layout_rank == 1:
        return [int(size(layout))]
    else:
        return [int(size(layout[i])) for i in range(layout_rank)]


def _is_hierarchical_2d(layout):
    """
    Check if a rank-2 layout has hierarchical structure.
    
    A layout is hierarchical if either mode has depth >= 1 (tuple structure).
    - depth 0 = simple integer
    - depth 1 = tuple (e.g., (2,2))
    - depth 2+ = nested tuple (e.g., ((2,2),3))
    
    For example: (2,(2,2)) or ((2,2),4)
    
    Args:
        layout: CuTe layout object
    
    Returns:
        True if hierarchical, False otherwise
    """
    layout_rank = int(rank(layout))
    
    if layout_rank != 2:
        return False
    
    # Check if either mode is hierarchical (depth >= 1 means tuple/nested)
    try:
        depth0 = int(depth(layout[0]))
        depth1 = int(depth(layout[1]))
        return depth0 >= 1 or depth1 >= 1
    except Exception as e:
        return False


def _extract_hierarchical_dimensions(layout, mode_idx, is_hierarchical, total_size):
    """
    Extract dimensions from a mode (hierarchical or simple).
    
    Returns:
        (rows, cols) tuple representing the grid structure of this mode
    """
    if is_hierarchical:
        # Try to extract (rows, cols) from hierarchical structure
        try:
            if rank(layout.shape[mode_idx]) == 2:
                dim0 = layout.shape[mode_idx][0]
                dim1 = layout.shape[mode_idx][1]
                rows = int(dim0) if isinstance(dim0, int) or hasattr(dim0, '__int__') else int(size(dim0))
                cols = int(dim1) if isinstance(dim1, int) or hasattr(dim1, '__int__') else int(size(dim1))
                return rows, cols
        except:
            pass
        # Fallback: try to factor total_size
        import math
        rows = cols = int(math.sqrt(total_size))
        if rows * cols != total_size:
            for r in range(int(math.sqrt(total_size)), 0, -1):
                if total_size % r == 0:
                    return r, total_size // r
        return rows, cols
    else:
        # Simple mode: arrange as a vertical column (or horizontal row)
        return total_size, 1


def _create_hierarchical_2d_layout_svg(layout):
    """
    Universal SVG generator for hierarchical 2D layouts.
    
    Handles all cases:
    - (simple, hierarchical): e.g., (2, (2,2))
    - (hierarchical, simple): e.g., ((2,2), 2)
    - (hierarchical, hierarchical): e.g., ((2,2), (3,4))
    
    Visualizes as a continuous grid with tile boundaries marked by thick blue lines.
    
    Args:
        layout: Hierarchical rank-2 CuTe layout
    
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
    
    # Determine which modes are hierarchical
    depth0 = int(depth(layout[0]))
    depth1 = int(depth(layout[1]))
    
    # Extract all indices
    indices_flat = _extract_layout_indices(layout)
    
    # Get total dimensions for each mode
    M, N = int(size(layout[0])), int(size(layout[1]))
    
    # Universal handling: Extract inner and outer dimensions
    # Mode 0 (leftmost) = INNER (what's inside each tile)
    # Mode 1 (rightmost) = OUTER (tile grid structure)
    
    inner_rows, inner_cols = _extract_hierarchical_dimensions(layout, 0, depth0 >= 1, M)
    tile_rows, tile_cols = _extract_hierarchical_dimensions(layout, 1, depth1 >= 1, N)
    
    # Reshape: (inner_rows, inner_cols, tile_rows, tile_cols)
    indices = indices_flat.reshape(inner_rows, inner_cols, tile_rows, tile_cols, order='F')
    
    # Universal visualization: continuous grid with tile boundaries
    label_margin = cell_size
    total_rows = inner_rows * tile_rows
    total_cols = inner_cols * tile_cols
    grid_width = total_cols * cell_size
    grid_height = total_rows * cell_size
    
    # Margins: left needs space for labels, top needs space for index labels, right/bottom minimal
    left_margin = 10 + label_margin
    top_margin = 25  # Enough space for two rows of labels (at -15 and -5)
    right_margin = 10
    bottom_margin = 10
    
    page_width = left_margin + grid_width + right_margin
    page_height = top_margin + grid_height + bottom_margin
    
    dwg = svgwrite.Drawing(size=(page_width, page_height))
    grid_start_x = left_margin
    grid_start_y = top_margin
    
    # Draw all cells in a continuous grid
    for tile_i in range(tile_rows):
        for tile_j in range(tile_cols):
            for inner_i in range(inner_rows):
                for inner_j in range(inner_cols):
                    abs_row = tile_i * inner_rows + inner_i
                    abs_col = tile_j * inner_cols + inner_j
                    idx = indices[inner_i, inner_j, tile_i, tile_j]
                    x = grid_start_x + abs_col * cell_size
                    y = grid_start_y + abs_row * cell_size
                    
                    dwg.add(dwg.rect(insert=(x, y), size=(cell_size, cell_size),
                                    fill=svgwrite.rgb(*rgb_255_colors[idx % len(rgb_255_colors)], mode="RGB"),
                                    stroke="black", stroke_width=0.5))
                    dwg.add(dwg.text(str(idx), insert=(x + cell_size // 2, y + cell_size // 2),
                                    text_anchor="middle", alignment_baseline="central", font_size="8px"))
    
    # Draw thick blue lines at tile boundaries
    for tile_i in range(tile_rows + 1):
        y = grid_start_y + tile_i * inner_rows * cell_size
        dwg.add(dwg.line(start=(grid_start_x, y), end=(grid_start_x + grid_width, y),
                       stroke="blue", stroke_width=2))
    for tile_j in range(tile_cols + 1):
        x = grid_start_x + tile_j * inner_cols * cell_size
        dwg.add(dwg.line(start=(x, grid_start_y), end=(x, grid_start_y + grid_height),
                       stroke="blue", stroke_width=2))
    
    # Labels: outer (tile indices) and inner (element indices within tiles)
    # Only show labels when the dimension is > 1 (otherwise it's redundant)
    
    # Top labels (columns)
    show_inner_cols = inner_cols > 1
    show_tile_cols = tile_cols > 1
    
    if show_inner_cols:
        # Show inner column indices (closer to grid)
        for tile_j in range(tile_cols):
            for inner_j in range(inner_cols):
                x = grid_start_x + (tile_j * inner_cols + inner_j) * cell_size + cell_size // 2
                dwg.add(dwg.text(str(inner_j), insert=(x, grid_start_y - 5),
                               text_anchor="middle", alignment_baseline="baseline", font_size="8px"))
    if show_tile_cols:
        # Show outer tile column indices (further from grid)
        for tile_j in range(tile_cols):
            x = grid_start_x + (tile_j * inner_cols + inner_cols / 2) * cell_size
            y_pos = grid_start_y - 15 if show_inner_cols else grid_start_y - 5
            dwg.add(dwg.text(str(tile_j), insert=(x, y_pos),
                           text_anchor="middle", alignment_baseline="baseline", font_size="10px", fill="blue"))
    
    # Left labels (rows)
    show_inner_rows = inner_rows > 1
    show_tile_rows = tile_rows > 1
    
    if show_tile_rows:
        # Show outer tile row indices (further from grid)
        for tile_i in range(tile_rows):
            y = grid_start_y + (tile_i * inner_rows + inner_rows / 2) * cell_size
            dwg.add(dwg.text(str(tile_i), insert=(grid_start_x - label_margin + 3, y),
                           text_anchor="start", alignment_baseline="central", font_size="10px", fill="blue"))
    if show_inner_rows:
        # Show inner row indices (closer to grid)
        for tile_i in range(tile_rows):
            for inner_i in range(inner_rows):
                y = grid_start_y + (tile_i * inner_rows + inner_i) * cell_size + cell_size // 2
                x_pos = grid_start_x - 8 if show_tile_rows else grid_start_x - label_margin + 3
                dwg.add(dwg.text(str(inner_i), insert=(x_pos, y),
                               text_anchor="middle" if show_tile_rows else "start", 
                               alignment_baseline="central", font_size="8px"))
    
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

    # Add margin for axis labels
    label_margin = cell_size
    page_width = N * cell_size + label_margin
    page_height = M * cell_size + label_margin

    filled = np.zeros((M, N), dtype=bool)
    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Draw white background grid (offset by label_margin)
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(j * cell_size + label_margin, i * cell_size + label_margin),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Draw colored cells with thread/value labels
    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords[tid, vid, 0]), int(coords[tid, vid, 1])
            dwg.add(
                dwg.rect(
                    insert=(j * cell_size + label_margin, i * cell_size + label_margin),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(*rgb_255_colors[tid % len(rgb_255_colors)], mode="RGB"),
                    stroke="black",
                )
            )
            text_label = f"T{tid}:V{vid}"
            dwg.add(
                dwg.text(
                    text_label,
                    insert=(j * cell_size + label_margin + cell_size // 2, i * cell_size + label_margin + cell_size // 2),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="6px",
                )
            )
            filled[i, j] = True

    # Check for any unfilled cells in the MxN domain
    unfilled_cells = []
    for i in range(M):
        for j in range(N):
            if not filled[i, j]:
                unfilled_cells.append((i, j))

    # Draw axis labels (offset by label_margin)
    for i in range(M):
        dwg.add(
            dwg.text(
                str(i),
                insert=(label_margin // 2, i * cell_size + label_margin + cell_size // 2),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="10px",
            )
        )

    for j in range(N):
        dwg.add(
            dwg.text(
                str(j),
                insert=(j * cell_size + label_margin + cell_size // 2, label_margin // 2),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="10px",
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

    # Add margin for axis labels
    label_margin = cell_size
    page_width = N * cell_size + label_margin
    page_height = M * cell_size + label_margin

    filled = np.zeros((M, N), dtype=bool)
    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Draw white background grid (offset by label_margin)
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(j * cell_size + label_margin, i * cell_size + label_margin),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Draw colored cells with thread/value labels
    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords[tid, vid, 0]), int(coords[tid, vid, 1])
            x = j * cell_size + label_margin
            y = i * cell_size + label_margin

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

    # Add axis labels (matching C++ print_latex_tv behavior)
    # Top labels: column indices (0 to N-1)
    for j in range(N):
        x = j * cell_size + label_margin + cell_size // 2
        y = label_margin // 2
        dwg.add(
            dwg.text(
                str(j),
                insert=(x, y),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
            )
        )

    # Left labels: row indices (0 to M-1)
    for i in range(M):
        x = label_margin // 2
        y = i * cell_size + label_margin + cell_size // 2
        dwg.add(
            dwg.text(
                str(i),
                insert=(x, y),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
            )
        )

    return dwg


def render_layout_svg(layout, output_file, flatten_hierarchical=True):
    """
    Render a CuTe layout as an SVG grid with color-coded cells.
    
    Supports layouts of any rank and structure:
    - 1D: e.g., 8:1 - horizontal bar
    - 2D: e.g., (4,8):(1,4) - 2D grid
    - 3D+: e.g., (2,4,8):(1,8,32) - visualized as horizontal slices
    - Hierarchical: e.g., (2,(2,2)):(4,(2,1)) - flat by default, or with tile boundaries

    Args:
        layout: CuTe layout object (any rank, any structure)
        output_file: Output SVG file path
        flatten_hierarchical: If True (default), hierarchical layouts are rendered as flat grids.
                             If False, hierarchical layouts are rendered with tile boundaries.
    """
    dwg = _create_layout_svg(layout, flatten_hierarchical=flatten_hierarchical)
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


def display_layout(layout, flatten_hierarchical=True):
    """
    Display a CuTe layout directly in Jupyter notebooks without writing to disk.
    
    Supports layouts of any rank and structure:
    - 1D: e.g., 8:1 - horizontal bar
    - 2D: e.g., (4,8):(1,4) - 2D grid
    - 3D+: e.g., (2,4,8):(1,8,32) - visualized as horizontal slices
    - Hierarchical: e.g., (2,(2,2)):(4,(2,1)) - flat by default, or with tile boundaries

    Args:
        layout: CuTe layout object (any rank, any structure)
        flatten_hierarchical: If True (default), hierarchical layouts are rendered as flat grids.
                             If False, hierarchical layouts are rendered with tile boundaries.

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    dwg = _create_layout_svg(layout, flatten_hierarchical=flatten_hierarchical)
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

    # Add margin for axis labels (1 cell on left, 1 cell on top)
    label_margin = cell_size
    page_width = N * cell_size + label_margin
    page_height = M * cell_size + label_margin

    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Draw grid cells (offset by label_margin)
    for i in range(M):
        for j in range(N):
            idx = indices[i, j]
            x = j * cell_size + label_margin
            y = i * cell_size + label_margin

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

    # Add axis labels (matching C++ print_latex behavior)
    # Top labels: column indices (0 to N-1)
    for j in range(N):
        x = j * cell_size + label_margin + cell_size // 2
        y = label_margin // 2
        dwg.add(
            dwg.text(
                str(j),
                insert=(x, y),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
            )
        )

    # Left labels: row indices (0 to M-1)
    for i in range(M):
        x = label_margin // 2
        y = i * cell_size + label_margin + cell_size // 2
        dwg.add(
            dwg.text(
                str(i),
                insert=(x, y),
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
    gap = 1 * cell_size

    # Add margin for axis labels (1 cell on left for S, 1 cell on top, 1 cell on right for D)
    label_margin = cell_size
    total_width = 2 * N * cell_size + gap + 2 * label_margin
    page_height = M * cell_size + label_margin

    filled_s = np.zeros((M, N), dtype=bool)
    filled_d = np.zeros((M, N), dtype=bool)
    dwg = svgwrite.Drawing(size=(total_width, page_height))

    # Draw source grid (left side) - background cells
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(label_margin + j * cell_size, label_margin + i * cell_size),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Draw destination grid (right side) - background cells
    x_offset = label_margin + N * cell_size + gap
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(x_offset + j * cell_size, label_margin + i * cell_size),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Fill source grid with thread-value data
    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords_s[tid, vid, 0]), int(coords_s[tid, vid, 1])
            x = label_margin + j * cell_size
            y = label_margin + i * cell_size

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
            y = label_margin + i * cell_size

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

    # Add axis labels for source grid (matching C++ print_latex_copy behavior)
    # Top labels: column indices (0 to N-1)
    for j in range(N):
        x = j * cell_size + label_margin + cell_size // 2
        y = label_margin // 2
        dwg.add(dwg.text(str(j), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

    # Left labels: row indices (0 to M-1)
    for i in range(M):
        x = label_margin // 2
        y = i * cell_size + label_margin + cell_size // 2
        dwg.add(dwg.text(str(i), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

    # Add axis labels for destination grid
    # Top labels: column indices (0 to N-1)
    for j in range(N):
        x = x_offset + j * cell_size + cell_size // 2
        y = label_margin // 2
        dwg.add(dwg.text(str(j), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

    # Right labels: row indices (0 to M-1) - placed on RIGHT side for D grid
    for i in range(M):
        x = x_offset + N * cell_size + label_margin // 2
        y = i * cell_size + label_margin + cell_size // 2
        dwg.add(dwg.text(str(i), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

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

    # Add margin for axis labels
    label_margin = 0

    # SVG dimensions
    page_width = (K + N + 2) * cell_size + label_margin
    page_height = (K + M + 2) * cell_size + label_margin

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

    # Colors (matching C++ SVGColor_TV from print_svg.hpp)
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

    # --- Draw C (M×N at bottom-right) ---
    for tid in range(num_threads_C):
        for vid in range(num_values_C):
            m, n = int(coords_C[tid, vid, 0]), int(coords_C[tid, vid, 1])
            if m < M and n < N and not filled[m, n, 0]:
                filled[m, n, 0] = True

                x = label_margin + (n + K + 2) * cell_size
                y = label_margin + (m + K + 2) * cell_size

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

                x = label_margin + (k + 1) * cell_size
                y = label_margin + (m + K + 2) * cell_size

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

                x = label_margin + (n + K + 2) * cell_size
                y = label_margin + (k + 1) * cell_size

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

    # Add axis labels (matching C++ print_latex_mma behavior)

    # --- A matrix (M×K) axis labels ---
    # Top labels: K dimension (0 to K-1)
    for k in range(K):
        x = label_margin + (k + 1) * cell_size + cell_size // 2
        y = label_margin + (K + 2) * cell_size - cell_size // 2
        dwg.add(dwg.text(str(k), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

    # Left labels: M dimension (0 to M-1)
    for m in range(M):
        x = label_margin + cell_size // 2
        y = label_margin + (m + K + 2) * cell_size + cell_size // 2
        dwg.add(dwg.text(str(m), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

    # --- B matrix (K×N, shown transposed) axis labels ---
    # Top labels: K dimension (0 to K-1)
    for k in range(K):
        x = label_margin + (K + 2) * cell_size - cell_size // 2
        y = label_margin + (k + 1) * cell_size + cell_size // 2
        dwg.add(dwg.text(str(k), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

    # Right labels: N dimension (0 to N-1)
    for n in range(N):
        x = label_margin + (n + K + 2) * cell_size + cell_size // 2
        y = label_margin + cell_size // 2
        dwg.add(dwg.text(str(n), insert=(x, y),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8px"))

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


def render_mma_from_layouts(layoutC, layoutA, layoutB, tile_mnk, output_file):
    """
    Render MMA layout from manually constructed TV layouts.

    Low-level API for custom layout visualization.

    Args:
        layoutC: C matrix TV layout (M×N)
        layoutA: A matrix TV layout (M×K)
        layoutB: B matrix TV layout (N×K)
        tile_mnk: Tuple (M, N, K) tile dimensions
        output_file: Output SVG file path
    """
    M, N, K = tile_mnk

    # Create identity tensors and compose
    refC = make_identity_tensor((M, N))
    tensorC_TV = cute.composition(refC, layoutC)

    refA = make_identity_tensor((M, K))
    tensorA_TV = cute.composition(refA, layoutA)

    refB = make_identity_tensor((N, K))
    tensorB_TV = cute.composition(refB, layoutB)

    # Handle potential extra dimensions
    tensorC = tensorC_TV[:, :, 0] if hasattr(tensorC_TV, 'ndim') and tensorC_TV.ndim > 2 else tensorC_TV
    tensorA = tensorA_TV[:, :, 0] if hasattr(tensorA_TV, 'ndim') and tensorA_TV.ndim > 2 else tensorA_TV
    tensorB = tensorB_TV[:, :, 0] if hasattr(tensorB_TV, 'ndim') and tensorB_TV.ndim > 2 else tensorB_TV

    # Create SVG using the same internal logic
    cell_size = 20
    page_width = (K + N + 2) * cell_size
    page_height = (K + M + 2) * cell_size

    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Track filled cells
    import numpy as np
    filled = np.zeros((M, N, K), dtype=bool)

    # Get sizes
    num_threads_C = size(tensorC, mode=[0])
    num_values_C = size(tensorC, mode=[1])
    num_threads_A = size(tensorA, mode=[0])
    num_values_A = size(tensorA, mode=[1])
    num_threads_B = size(tensorB, mode=[0])
    num_values_B = size(tensorB, mode=[1])

    # Extract coordinates
    coords_C, coords_A, coords_B = _extract_mma_coords(
        tensorC, tensorA, tensorB,
        num_threads_C, num_values_C,
        num_threads_A, num_values_A,
        num_threads_B, num_values_B
    )

    # Colors (matching C++ SVGColor_TV from print_svg.hpp)
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

    # Draw C
    for tid in range(num_threads_C):
        for vid in range(num_values_C):
            m, n = int(coords_C[tid, vid, 0]), int(coords_C[tid, vid, 1])
            if m < M and n < N and not filled[m, n, 0]:
                filled[m, n, 0] = True
                x = (n + K + 2) * cell_size
                y = (m + K + 2) * cell_size
                color = rgb_255_colors[tid % len(rgb_255_colors)]
                rect = dwg.rect(insert=(x, y), size=(cell_size, cell_size),
                               fill=svgwrite.rgb(*color, mode="RGB"), stroke='black')
                dwg.add(rect)
                text1 = dwg.text(f'T{tid}', insert=(x + cell_size/2, y + cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text1)
                text2 = dwg.text(f'V{vid}', insert=(x + cell_size/2, y + 3*cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text2)

    filled.fill(False)

    # Draw A
    for tid in range(num_threads_A):
        for vid in range(num_values_A):
            m, k = int(coords_A[tid, vid, 0]), int(coords_A[tid, vid, 1])
            if m < M and k < K and not filled[m, 0, k]:
                filled[m, 0, k] = True
                x = (k + 1) * cell_size
                y = (m + K + 2) * cell_size
                color = rgb_255_colors[tid % len(rgb_255_colors)]
                rect = dwg.rect(insert=(x, y), size=(cell_size, cell_size),
                               fill=svgwrite.rgb(*color, mode="RGB"), stroke='black')
                dwg.add(rect)
                text1 = dwg.text(f'T{tid}', insert=(x + cell_size/2, y + cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text1)
                text2 = dwg.text(f'V{vid}', insert=(x + cell_size/2, y + 3*cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text2)

    filled.fill(False)

    # Draw B
    for tid in range(num_threads_B):
        for vid in range(num_values_B):
            n, k = int(coords_B[tid, vid, 0]), int(coords_B[tid, vid, 1])
            if n < N and k < K and not filled[0, n, k]:
                filled[0, n, k] = True
                x = (n + K + 2) * cell_size
                y = (k + 1) * cell_size
                color = rgb_255_colors[tid % len(rgb_255_colors)]
                rect = dwg.rect(insert=(x, y), size=(cell_size, cell_size),
                               fill=svgwrite.rgb(*color, mode="RGB"), stroke='black')
                dwg.add(rect)
                text1 = dwg.text(f'T{tid}', insert=(x + cell_size/2, y + cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text1)
                text2 = dwg.text(f'V{vid}', insert=(x + cell_size/2, y + 3*cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text2)

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
    return display(SVG(dwg.tostring()))


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
    from IPython.display import SVG, display
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    return display(SVG(dwg.tostring()))
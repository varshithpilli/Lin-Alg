import numpy as np
from math import gcd
from . import utils

def row_echelon_form_steps(matrix):
    """
    Compute the Row Echelon Form (REF) of a matrix using integer-preserving row operations.
    Uses A[i] = m * A[i] - n * A[r] with minimized m, n (divided by GCD) and positive m.
    Minimizes row swaps, does not enforce staircase pattern, and returns step-by-step details.
    Returns a list of steps, each with description, matrix state, pivot coordinates, and all pivots.
    Uses 1-based indexing in descriptions for user-friendliness.
    """
    try:
        A = np.array(matrix, dtype=float)
    except ValueError as e:
        raise ValueError(f"Invalid input matrix: {e}. Ensure it's a list of lists of numbers.")

    if A.size == 0:
        raise ValueError("Input matrix cannot be empty.")
    if len(A.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")

    rows, cols = A.shape
    r = 0
    steps = []
    current_pivots = []
    current_step_num = 1

    steps.append({
        'description': f'<b>Step {current_step_num}: Initial Matrix</b>',
        'matrix': utils.format_matrix(A),
        'pivot_coord': None,
        'all_pivot_coords': []
    })
    current_step_num += 1

    A_copy = A.copy()

    while r < rows:
        pivot_col = -1
        for j in range(cols):
            if abs(A_copy[r, j]) > 1e-10:
                pivot_col = j
                break

        if pivot_col == -1:
            steps.append({
                'description': f'<b>Step {current_step_num}: Row {r+1} Analysis</b><br>Row {r+1} is all zeros. No pivot in this row. Skipping to next row.',
                'matrix': utils.format_matrix(A_copy),
                'pivot_coord': None,
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
            r += 1
            continue

        pivot_val = A_copy[r, pivot_col]
        current_pivot_coord = [r, pivot_col]
        if current_pivot_coord not in current_pivots:
            current_pivots.append(current_pivot_coord)

        steps.append({
            'description': f'<b>Step {current_step_num}: Identify Pivot in Row {r+1}</b><br>Pivot found at [{r+1},{pivot_col+1}] with value {pivot_val:.4f}. Using this pivot to eliminate entries below it.',
            'matrix': utils.format_matrix(A_copy),
            'pivot_coord': [r, pivot_col],
            'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
        })
        current_step_num += 1

        elim_header = f'<b>Step {current_step_num}: Eliminate Below Pivot [{r+1},{pivot_col+1}]</b>'
        elim_details = []
        elim_occurred = False
        matrix_changed_in_elim = False
        temp_A_before_elim = A_copy.copy()

        for i in range(r + 1, rows):
            if abs(A_copy[i, pivot_col]) > 1e-10:
                elim_occurred = True
                m = A_copy[r, pivot_col]
                n = A_copy[i, pivot_col]
                m_int = int(round(m))
                n_int = int(round(n))
                g = 0
                if abs(m - m_int) < 1e-9 and abs(n - n_int) < 1e-9 and m_int != 0 and n_int != 0:
                    try:
                        g = gcd(abs(m_int), abs(n_int))
                    except TypeError:
                        g = 0
                if g > 1:
                    m = m / g
                    n = n / g
                if m < 0:
                    m = -m
                    n = -n
                m_display = m
                n_display = n
                op_desc = f'  Target: R<sub>{i+1}</sub>, Pivot: R<sub>{r+1}</sub>. <span class="formula">R<sub>{i+1}</sub> → {m_display:.4f} * R<sub>{i+1}</sub> - ({n_display:.4f}) * R<sub>{r+1}</sub></span>'
                A_copy[i] = m * A_copy[i] - n * A_copy[r]
                A_copy[i, pivot_col] = 0.0
                elim_details.append(op_desc)
                matrix_changed_in_elim = True

        if elim_occurred and matrix_changed_in_elim:
            steps.append({
                'description': elim_header + "<br>" + "<br>".join(elim_details),
                'matrix': utils.format_matrix(A_copy),
                'pivot_coord': [r, pivot_col],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
        elif not elim_occurred and pivot_col != -1:
            steps.append({
                'description': f'<b>Step {current_step_num}: Elimination Check for Pivot [{r+1},{pivot_col+1}]</b><br>No non-zero elements found below the pivot in column {pivot_col+1}. No elimination needed for this pivot.',
                'matrix': utils.format_matrix(A_copy),
                'pivot_coord': [r, pivot_col],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1

        r += 1

    if np.all(np.abs(A_copy - np.round(A_copy)) < 1e-10):
        A_copy = np.round(A_copy).astype(int)
    final_A_display = utils.format_matrix(A_copy)

    steps.append({
        'description': f'<b>Step {current_step_num}: Final Row Echelon Form (REF) Reached</b><br>All rows processed. Pivots are identified at positions (1-indexed): {[[p[0]+1, p[1]+1] for p in current_pivots]}',
        'matrix': final_A_display,
        'pivot_coord': None,
        'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
    })

    return steps
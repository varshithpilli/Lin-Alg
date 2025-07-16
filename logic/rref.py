import numpy as np
from math import gcd
from . import utils

def row_reduced_echelon_form_steps(matrix):
    """
    Compute the Reduced Row Echelon Form (RREF) of a matrix using integer-preserving row operations,
    returning a list of steps for visualization.
    Ensures each pivot is 1, with zeros above and below it.
    Uses A[i] = m * A[i] - n * A[r] where needed.
    Returns steps including descriptions, matrix states, and pivot info.
    Uses 1-based indexing in descriptions.
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

    A_copy = A.copy()

    steps.append({
        'description': f'<b>Step {current_step_num}: Initial Matrix</b>',
        'matrix': utils.format_matrix(A_copy),
        'pivot_coord': None,
        'all_pivot_coords': []
    })
    current_step_num += 1

    processed_pivot_cols = set()
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        pivot_idx = -1
        for i in range(pivot_row, rows):
            if abs(A_copy[i, col]) > 1e-10:
                pivot_idx = i
                break
                
        if pivot_idx == -1:
            continue
            
        if pivot_idx != pivot_row:
            A_copy[[pivot_idx, pivot_row]] = A_copy[[pivot_row, pivot_idx]]
            steps.append({
                'description': f'<b>Step {current_step_num}: Row Swap</b><br>Swap R<sub>{pivot_idx+1}</sub> and R<sub>{pivot_row+1}</sub> to bring pivot candidate to Row {pivot_row+1}.',
                'matrix': utils.format_matrix(A_copy),
                'pivot_coord': [pivot_row, col],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots] 
            })
            current_step_num += 1
            
        pivot_val = A_copy[pivot_row, col]
        current_pivot_coord = [pivot_row, col]
        if current_pivot_coord not in current_pivots:
            current_pivots.append(current_pivot_coord)
            processed_pivot_cols.add(col)
            
        steps.append({
            'description': f'<b>Step {current_step_num}: Identify Pivot</b><br>Pivot identified at [{pivot_row+1},{col+1}] with value {pivot_val:.4f}.',
            'matrix': utils.format_matrix(A_copy),
            'pivot_coord': [pivot_row, col],
            'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
        })
        current_step_num += 1
            
        if abs(pivot_val - 1.0) > 1e-10:
            norm_desc = f'<b>Step {current_step_num}: Normalize Pivot Row {pivot_row + 1}</b><br>Divide R<sub>{pivot_row + 1}</sub> by pivot {pivot_val:.4f}. <span class="formula">R<sub>{pivot_row+1}</sub> → R<sub>{pivot_row+1}</sub> / {pivot_val:.4f}</span>'
            A_copy[pivot_row] = A_copy[pivot_row] / pivot_val
            A_copy[pivot_row, col] = 1.0
            steps.append({
                'description': norm_desc,
                'matrix': utils.format_matrix(A_copy),
                'pivot_coord': [pivot_row, col],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
            pivot_val = 1.0

        elim_below_header = f'<b>Step {current_step_num}: Eliminate Below Pivot [{pivot_row+1},{col+1}]</b>'
        elim_below_details = []
        elim_below_occurred = False
        for i in range(pivot_row + 1, rows):
            factor = A_copy[i, col]
            if abs(factor) > 1e-10:
                op_desc = f'&nbsp;&nbsp;<span class="formula">R<sub>{i+1}</sub> → R<sub>{i+1}</sub> - ({factor:.4f}) * R<sub>{pivot_row+1}</sub></span>'
                A_copy[i] -= factor * A_copy[pivot_row]
                A_copy[i, col] = 0.0
                elim_below_details.append(op_desc)
                elim_below_occurred = True
        
        if elim_below_occurred:
            steps.append({
                'description': elim_below_header + "<br>" + "<br>".join(elim_below_details),
                'matrix': utils.format_matrix(A_copy),
                'pivot_coord': [pivot_row, col],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
        elif pivot_val != 0:
             steps.append({
                 'description': f'<b>Step {current_step_num}: Elimination Check Below Pivot [{pivot_row+1},{col+1}]</b><br>All elements below pivot are already zero. No elimination needed below this pivot.',
                 'matrix': utils.format_matrix(A_copy),
                 'pivot_coord': [pivot_row, col],
                 'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
             })
             current_step_num += 1

        pivot_row += 1

    for p_idx in range(len(current_pivots) - 1, -1, -1):
        pivot_r, pivot_c = current_pivots[p_idx]
        
        elim_above_header = f'<b>Step {current_step_num}: Eliminate Above Pivot [{pivot_r+1},{pivot_c+1}]</b>'
        elim_above_details = []
        elim_above_occurred = False
        
        for i in range(pivot_r - 1, -1, -1): 
            factor = A_copy[i, pivot_c]
            if abs(factor) > 1e-10:
                op_desc = f'&nbsp;&nbsp;<span class="formula">R<sub>{i+1}</sub> → R<sub>{i+1}</sub> - ({factor:.4f}) * R<sub>{pivot_r+1}</sub></span>'
                A_copy[i] -= factor * A_copy[pivot_r]
                A_copy[i, pivot_c] = 0.0
                elim_above_details.append(op_desc)
                elim_above_occurred = True
                
        if elim_above_occurred:
            steps.append({
                'description': elim_above_header + "<br>" + "<br>".join(elim_above_details),
                'matrix': utils.format_matrix(A_copy),
                'pivot_coord': [pivot_r, pivot_c],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
        elif pivot_r > 0:
             steps.append({
                 'description': f'<b>Step {current_step_num}: Elimination Check Above Pivot [{pivot_r+1},{pivot_c+1}]</b><br>All elements above pivot are already zero. No elimination needed above this pivot.',
                 'matrix': utils.format_matrix(A_copy),
                 'pivot_coord': [pivot_r, pivot_c],
                 'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
             })
             current_step_num += 1
             
        
    final_A_display = utils.format_matrix(A_copy)
        
    steps.append({
        'description': f'<b>Step {current_step_num}: Final Reduced Row Echelon Form (RREF) Reached</b><br>All pivots are 1 and are the only non-zero entries in their columns. Pivots are at positions (1-indexed): {[[p[0]+1, p[1]+1] for p in current_pivots]}',
        'matrix': final_A_display,
        'pivot_coord': None,
        'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
    })

    return steps
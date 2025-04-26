import numpy as np
from math import gcd

def format_matrix(matrix):
    """Helper to format numpy matrix for JSON serialization, handling potential near-zero floats."""
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    return [[float(0.0) if abs(x) < 1e-10 else float(x) for x in row] for row in matrix]

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
    r = 0  # Current pivot row index (0-based)
    steps = []
    current_pivots = []
    current_step_num = 1

    A_copy = A.copy()

    # --- Initial Step --- 
    steps.append({
        'description': f'<b>Step {current_step_num}: Initial Matrix</b>',
        'matrix': format_matrix(A_copy),
        'pivot_coord': None,
        'all_pivot_coords': []
    })
    current_step_num += 1

    processed_pivot_cols = set() # Track columns already containing a processed pivot
    pivot_row = 0 # Tracks the target row for the next pivot
    
    # Forward elimination (similar to REF, but might normalize pivots)
    for col in range(cols):
        if pivot_row >= rows:
            break # All rows have pivots or are zero

        # Find a potential pivot in the current column at or below the pivot_row
        pivot_idx = -1
        for i in range(pivot_row, rows):
            if abs(A_copy[i, col]) > 1e-10:
                pivot_idx = i
                break
                
        if pivot_idx == -1:
            # No pivot in this column, move to the next
            continue
            
        # --- Row Swap (if needed) --- 
        if pivot_idx != pivot_row:
            A_copy[[pivot_idx, pivot_row]] = A_copy[[pivot_row, pivot_idx]]
            steps.append({
                'description': f'<b>Step {current_step_num}: Row Swap</b><br>Swap R<sub>{pivot_idx+1}</sub> and R<sub>{pivot_row+1}</sub> to bring pivot candidate to Row {pivot_row+1}.',
                'matrix': format_matrix(A_copy),
                'pivot_coord': [pivot_row, col], # Show the element that moved into pivot position
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots] 
            })
            current_step_num += 1
            
        # Current pivot identified at [pivot_row, col]
        pivot_val = A_copy[pivot_row, col]
        current_pivot_coord = [pivot_row, col]
        if current_pivot_coord not in current_pivots:
            current_pivots.append(current_pivot_coord)
            processed_pivot_cols.add(col)
            
        steps.append({
            'description': f'<b>Step {current_step_num}: Identify Pivot</b><br>Pivot identified at [{pivot_row+1},{col+1}] with value {pivot_val:.4f}.',
            'matrix': format_matrix(A_copy),
            'pivot_coord': [pivot_row, col],
            'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
        })
        current_step_num += 1
            
        # --- Normalize Pivot Row --- 
        if abs(pivot_val - 1.0) > 1e-10:
            norm_desc = f'<b>Step {current_step_num}: Normalize Pivot Row {pivot_row + 1}</b><br>Divide R<sub>{pivot_row + 1}</sub> by pivot {pivot_val:.4f}. <span class="formula">R<sub>{pivot_row+1}</sub> → R<sub>{pivot_row+1}</sub> / {pivot_val:.4f}</span>'
            A_copy[pivot_row] = A_copy[pivot_row] / pivot_val
            A_copy[pivot_row, col] = 1.0 # Ensure exactly 1
            steps.append({
                'description': norm_desc,
                'matrix': format_matrix(A_copy),
                'pivot_coord': [pivot_row, col],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
            pivot_val = 1.0 # Pivot is now 1

        # --- Eliminate Below Pivot --- 
        elim_below_header = f'<b>Step {current_step_num}: Eliminate Below Pivot [{pivot_row+1},{col+1}]</b>'
        elim_below_details = []
        elim_below_occurred = False
        for i in range(pivot_row + 1, rows):
            factor = A_copy[i, col]
            if abs(factor) > 1e-10:
                op_desc = f'&nbsp;&nbsp;<span class="formula">R<sub>{i+1}</sub> → R<sub>{i+1}</sub> - ({factor:.4f}) * R<sub>{pivot_row+1}</sub></span>'
                A_copy[i] -= factor * A_copy[pivot_row]
                A_copy[i, col] = 0.0 # Ensure exactly 0
                elim_below_details.append(op_desc)
                elim_below_occurred = True
        
        if elim_below_occurred:
            steps.append({
                'description': elim_below_header + "<br>" + "<br>".join(elim_below_details),
                'matrix': format_matrix(A_copy),
                'pivot_coord': [pivot_row, col],
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
        elif pivot_val != 0: # If pivot existed but nothing below needed clearing
             steps.append({
                 'description': f'<b>Step {current_step_num}: Elimination Check Below Pivot [{pivot_row+1},{col+1}]</b><br>All elements below pivot are already zero. No elimination needed below this pivot.',
                 'matrix': format_matrix(A_copy),
                 'pivot_coord': [pivot_row, col],
                 'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
             })
             current_step_num += 1

        pivot_row += 1 # Move to the next row to look for the next pivot

    # --- Backward Elimination (Clear Above Pivots) --- 
    # Iterate through pivots found (in reverse order of finding them is often efficient)
    for p_idx in range(len(current_pivots) - 1, -1, -1):
        pivot_r, pivot_c = current_pivots[p_idx]
        
        elim_above_header = f'<b>Step {current_step_num}: Eliminate Above Pivot [{pivot_r+1},{pivot_c+1}]</b>'
        elim_above_details = []
        elim_above_occurred = False
        
        # Check rows above the pivot
        for i in range(pivot_r - 1, -1, -1): 
            factor = A_copy[i, pivot_c]
            if abs(factor) > 1e-10:
                op_desc = f'&nbsp;&nbsp;<span class="formula">R<sub>{i+1}</sub> → R<sub>{i+1}</sub> - ({factor:.4f}) * R<sub>{pivot_r+1}</sub></span>'
                A_copy[i] -= factor * A_copy[pivot_r]
                A_copy[i, pivot_c] = 0.0 # Ensure exact zero
                elim_above_details.append(op_desc)
                elim_above_occurred = True
                
        if elim_above_occurred:
            steps.append({
                'description': elim_above_header + "<br>" + "<br>".join(elim_above_details),
                'matrix': format_matrix(A_copy),
                'pivot_coord': [pivot_r, pivot_c], # Highlight the pivot used for elimination
                'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
            })
            current_step_num += 1
        elif pivot_r > 0: # Only add check step if there were rows above
             steps.append({
                 'description': f'<b>Step {current_step_num}: Elimination Check Above Pivot [{pivot_r+1},{pivot_c+1}]</b><br>All elements above pivot are already zero. No elimination needed above this pivot.',
                 'matrix': format_matrix(A_copy),
                 'pivot_coord': [pivot_r, pivot_c],
                 'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
             })
             current_step_num += 1
             
    # --- Final Step --- 
    # Convert back to integers if all entries are effectively integers
    # This might obscure fractions in the final RREF, so optional
    # if np.all(np.abs(A_copy - np.round(A_copy)) < 1e-10):
    #     A_copy = np.round(A_copy).astype(int)
        
    final_A_display = format_matrix(A_copy)
        
    steps.append({
        'description': f'<b>Step {current_step_num}: Final Reduced Row Echelon Form (RREF) Reached</b><br>All pivots are 1 and are the only non-zero entries in their columns. Pivots are at positions (1-indexed): {[[p[0]+1, p[1]+1] for p in current_pivots]}',
        'matrix': final_A_display,
        'pivot_coord': None,
        'all_pivot_coords': [[p[0], p[1]] for p in current_pivots]
    })

    return steps

# Test function adapted for the new steps structure
if __name__ == "__main__":
    m1 = [
        [1, 2, 0, 1],
        [0, -1, 2, 1],
        [0, 0, 0, 2],
        [2, 3, 2, 3] # Add another row
    ]
    # m1 = [[0, 1, 2], [1, 0, 3], [4, 5, 6]] # Test case needing swaps
    # m1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]] # Test dependent rows
    
    print("Original Matrix:")
    print(np.array(m1))
    print("\n--- STEPS ---")
    
    try:
        rref_steps = row_reduced_echelon_form_steps(m1)
        for i, step in enumerate(rref_steps):
            print(f"\nStep {i+1}: {step['description']}")
            print("Matrix:")
            print(np.array(step['matrix']))
            if step.get('pivot_coord'):
                print(f"Pivot for this step: {step['pivot_coord']}")
            if step.get('all_pivot_coords'):
                print(f"All pivots identified so far: {step['all_pivot_coords']}")
                
    except ValueError as e:
        print(f"Error: {e}")
import numpy as np

def format_matrix(mat):
    if not isinstance(mat, np.ndarray):
        try: mat = np.array(mat, dtype=float) 
        except (ValueError, TypeError): return []
    if mat.size == 0: return []
    if np.iscomplexobj(mat):
        return [[{'real': round(val.real, 4), 'imag': round(val.imag, 4)} for val in row] for row in mat.tolist()]
    else:
        return [[round(float(val), 4) for val in row] for row in mat.tolist()]

def inverse_gauss_jordan(A):
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    n = A.shape[0]
    I = np.identity(n)
    
    # Create augmented matrix [A | I]
    augmented_matrix = np.hstack((A, I))
    steps = []
    current_step_num = 1

    # Add initial step
    steps.append({
        'description': f'<b>Step {current_step_num}: Start with Augmented Matrix [A | I]</b>',
        'matrix': format_matrix(augmented_matrix),
        'split_col_index': n # Index after the original A matrix part
    })
    current_step_num += 1

    # --- Forward Elimination --- 
    for col in range(n): # Process columns 0 to n-1 (original A part)
        # Find pivot row at or below current row `col`
        pivot_row = col
        while pivot_row < n and abs(augmented_matrix[pivot_row, col]) < 1e-10:
            pivot_row += 1
        
        # Check if pivot is found
        if pivot_row == n:
            raise ValueError("Matrix is singular and cannot be inverted (column of zeros found).")
            
        # Swap rows if necessary
        if pivot_row != col:
            swap_desc = f'<b>Step {current_step_num}: Row Swap</b><br>Swap R<sub>{col + 1}</sub> ↔ R<sub>{pivot_row + 1}</sub> to bring pivot to Row {col + 1}.'
            augmented_matrix[[col, pivot_row]] = augmented_matrix[[pivot_row, col]]
            steps.append({
                'description': swap_desc,
                'matrix': format_matrix(augmented_matrix),
                'split_col_index': n
            })
            current_step_num += 1
            
        # Normalize the pivot row
        pivot_val = augmented_matrix[col, col]
        if abs(pivot_val - 1.0) > 1e-10: # Avoid division if already 1
            norm_desc = f'<b>Step {current_step_num}: Normalize Pivot Row {col + 1}</b><br>Divide R<sub>{col + 1}</sub> by pivot {pivot_val:.4f}. <span class="formula">R<sub>{col+1}</sub> → R<sub>{col+1}</sub> / {pivot_val:.4f}</span>'
            augmented_matrix[col] = augmented_matrix[col] / pivot_val
            augmented_matrix[col, col] = 1.0 # Ensure exactly 1
            steps.append({
                'description': norm_desc,
                'matrix': format_matrix(augmented_matrix),
                'split_col_index': n,
                'pivot_coord': [col, col] # Highlight the pivot
            })
            current_step_num += 1
            
        # Eliminate other entries in the pivot column
        elim_occurred = False
        elim_header = f'<b>Step {current_step_num}: Eliminate in Column {col+1} using Pivot Row {col+1}</b>'
        elim_details = []
        for i in range(n): # Eliminate below and above in inverse calc
            if i != col:
                factor = augmented_matrix[i, col]
                if abs(factor) > 1e-10:
                    op_desc = f'&nbsp;&nbsp;<span class="formula">R<sub>{i+1}</sub> → R<sub>{i+1}</sub> - ({factor:.4f}) * R<sub>{col+1}</sub></span>'
                    augmented_matrix[i] -= factor * augmented_matrix[col]
                    augmented_matrix[i, col] = 0.0 # Ensure exactly 0
                    elim_details.append(op_desc)
                    elim_occurred = True
        
        if elim_occurred:
            steps.append({
                'description': elim_header + "<br>" + "<br>".join(elim_details),
                'matrix': format_matrix(augmented_matrix),
                'split_col_index': n,
                'pivot_coord': [col, col] # Highlight pivot used for elimination
            })
            current_step_num += 1

    # --- Backward Elimination (already done during forward pass for inverse) --- 
    # The loop above brings the left side to identity matrix directly.

    # Final Step: Extract Inverse
    inverse_matrix = augmented_matrix[:, n:] # Right half is the inverse
    steps.append({
        'description': f'<b>Step {current_step_num}: Reduced Row Echelon Form [I | A<sup>-1</sup>] Reached</b><br>The right side is the inverse matrix A<sup>-1</sup>.',
        'matrix': format_matrix(augmented_matrix),
        'split_col_index': n,
        'final_inverse': format_matrix(inverse_matrix) # Store final inverse separately if needed
    })
    
    return steps

def main():
    A = np.array([
        [4, 7, 2],
        [3, 9, 6],
        [8, 1, 5]
    ])
    # A = np.array([[1, 2], [2, 4]]) # Test singular matrix
    # A = np.array([[0, 1], [1, 0]]) # Test swap needed matrix

    try:
        steps = inverse_gauss_jordan(A)
        print("\nSteps for Gauss-Jordan Inversion:")
        for i, step in enumerate(steps):
            print(f"\nStep {i+1}: {step['description']}")
            print(np.array(step['matrix']))
            if 'final_inverse' in step:
                print("\nFinal Inverse matrix:")
                print(np.array(step['final_inverse']))

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

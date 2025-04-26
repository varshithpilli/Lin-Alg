import numpy as np
from math import gcd
# Import the RREF steps function from rref.py
from rref import row_reduced_echelon_form_steps, format_matrix

def gauss_jordan_solver(augmented_matrix, verbose=True):
    """
    Solve a system of linear equations (homogeneous or non-homogeneous) using Gauss-Jordan elimination.
    Computes the RREF of the augmented matrix and extracts solutions for the variables.
    Uses integer-preserving operations with minimized m, n (divided by GCD) and positive m.
    Returns a dictionary with RREF, solution type, and variable values or solution space description.
    If verbose=True, prints step-by-step operations using 1-based indexing.
    """
    # Convert input to numpy array
    A = np.array(augmented_matrix, dtype=float)
    rows, cols = A.shape
    num_vars = cols - 1  # Number of variables (excluding the constant column)
    
    if verbose:
        print("Initial Augmented Matrix:")
        print(A)
        print()

    # Initialize variables
    r = 0  # Current row
    step = 0  # Step counter
    pivot_columns = []  # Track columns with pivots

    # Gauss-Jordan Elimination (RREF)
    while r < rows:
        pivot_col = -1
        for j in range(cols - 1):  # Exclude constant column for pivots
            if abs(A[r, j]) > 1e-10 and j not in pivot_columns:
                pivot_col = j
                break

        if pivot_col == -1:
            # No pivot in this row, check for inconsistency
            if abs(A[r, cols-1]) > 1e-10:  # Row like [0, 0, ..., 0 | c], c != 0
                if verbose:
                    print("System is inconsistent (non-zero constant in zero row).")
                    print("Final Matrix:")
                    print(A)
                    print()
                return {
                    "rref": A.tolist(),
                    "solution_type": "inconsistent",
                    "solution": "No solution"
                }
            if verbose:
                step += 1
                print(f"Step {step}: Row {r+1} has no pivot, skipping to next row.")
                print("Matrix:")
                print(A)
                print()
            r += 1
            continue

        pivot_columns.append(pivot_col)
        pivot_val = A[r, pivot_col]

        # Normalize pivot to 1
        if abs(pivot_val - 1.0) > 1e-10:
            step += 1
            if verbose:
                print(f"Step {step}: Normalize pivot A[{r+1},{pivot_col+1}]={pivot_val} to 1")
                print(f"Operation: R[{r+1}] = R[{r+1}] / {pivot_val}")
            A[r] = A[r] / pivot_val
            if verbose:
                print("Matrix:")
                print(A)
                print()

        # Eliminate elements above and below the pivot
        for i in range(rows):
            if i != r and abs(A[i, pivot_col]) > 1e-10:
                m = A[r, pivot_col]  # Should be 1
                n = A[i, pivot_col]
                m_int = int(round(m))
                n_int = int(round(n))
                g = gcd(abs(m_int), abs(n_int)) if m_int != 0 and n_int != 0 else 1
                if g > 1:
                    m = m / g
                    n = n / g
                if m < 0:
                    m = -m
                    n = -n
                step += 1
                if verbose:
                    print(f"Step {step}: Eliminate A[{i+1},{pivot_col+1}]={A[i, pivot_col]} using pivot A[{r+1},{pivot_col+1}]={A[r, pivot_col]}")
                    print(f"Operation: R[{i+1}] = {m} * R[{i+1}] - {n} * R[{r+1}]")
                A[i] = m * A[i] - n * A[r]
                A[i, pivot_col] = 0.0
                if verbose:
                    print("Matrix:")
                    print(A)
                    print()

        r += 1

    # Convert to integers if possible
    if np.all(np.abs(A - np.round(A)) < 1e-10):
        A = np.round(A).astype(int)

    # Analyze the RREF for solutions
    rank = len(pivot_columns)
    is_homogeneous = np.all(np.abs(A[:, -1]) < 1e-10)  # Check if b = 0

    if verbose:
        print("Final Reduced Row Echelon Form:")
        print(A)
        print()

    # Check for inconsistency
    for i in range(rows):
        if all(abs(A[i, j]) < 1e-10 for j in range(cols-1)) and abs(A[i, cols-1]) > 1e-10:
            return {
                "rref": A.tolist(),
                "solution_type": "inconsistent",
                "solution": "No solution"
            }

    # Determine solution type and extract solutions
    if rank == num_vars and rank <= rows:
        # Unique solution
        solution = [A[i, cols-1] for i in range(min(rows, num_vars)) if i < len(pivot_columns)]
        solution += [0.0] * (num_vars - len(solution))  # Pad with zeros for underdetermined systems
        solution_dict = {f"x{i+1}": val for i, val in enumerate(solution)}
        return {
            "rref": A.tolist(),
            "solution_type": "unique",
            "solution": solution_dict
        }
    else:
        # Infinite solutions (free variables)
        free_vars = [i for i in range(num_vars) if i not in pivot_columns]
        solution_desc = "Infinite solutions with free variables: "
        solution_desc += ", ".join([f"x{i+1}" for i in free_vars]) if free_vars else "None (trivial solution)"
        
        # For homogeneous systems, describe the solution space
        if is_homogeneous:
            if free_vars:
                solution_desc += "\nNon-trivial solutions exist. General solution involves linear combinations of free variables."
            else:
                solution_desc = "Trivial solution: " + ", ".join([f"x{i+1}=0" for i in range(num_vars)])
        else:
            # Non-homogeneous: Provide particular solution and free variables
            particular = [0.0] * num_vars
            for i, col in enumerate(pivot_columns):
                if i < rows:
                    particular[col] = A[i, cols-1]
            solution_desc += f"\nParticular solution: {', '.join([f'x{i+1}={val}' for i, val in enumerate(particular)])}"
            if free_vars:
                solution_desc += "\nGeneral solution: Particular solution + linear combinations of free variables."

        return {
            "rref": A.tolist(),
            "solution_type": "infinite",
            "solution": solution_desc
        }

def solve_equations_steps(augmented_matrix):
    """
    Solves a system of linear equations from an augmented matrix by first computing RREF steps
    and then interpreting the result.
    Returns a list of steps including RREF calculation and solution analysis.
    """
    try:
        A_orig = np.array(augmented_matrix, dtype=float)
        if len(A_orig.shape) != 2 or A_orig.shape[1] < 2:
            raise ValueError("Input must be a 2D augmented matrix with at least 2 columns.")
    except ValueError as e:
        raise ValueError(f"Invalid input matrix: {e}. Ensure it's a list of lists of numbers.")

    # --- Step 1: Get RREF Steps --- 
    # We directly call the RREF steps function which includes its own validation
    rref_steps = row_reduced_echelon_form_steps(augmented_matrix)
    
    # The last step from rref_steps contains the final RREF matrix and pivot info
    final_rref_step = rref_steps[-1]
    A_rref = np.array(final_rref_step['matrix'], dtype=float)
    rows, cols = A_rref.shape
    num_vars = cols - 1
    pivots = final_rref_step['all_pivot_coords'] # List of [row, col] for pivots
    pivot_cols = sorted([p[1] for p in pivots]) # Get just the column indices of pivots
    rank = len(pivot_cols)
    current_step_num = len(rref_steps) + 1 # Continue step numbering

    # --- Step 2: Analyze RREF for Consistency --- 
    inconsistent = False
    inconsistency_reason = ""
    for r in range(rows):
        row_is_zero_coeffs = all(abs(A_rref[r, c]) < 1e-10 for c in range(num_vars))
        if row_is_zero_coeffs and abs(A_rref[r, -1]) > 1e-10: # Check constant term B[r]
            inconsistent = True
            inconsistency_reason = f"Row {r+1} has the form [0 0 ... 0 | c] where c = {A_rref[r, -1]:.4f} ≠ 0."
            break
            
    analysis_description = f"<b>Step {current_step_num}: Analyze RREF for Consistency</b><br>" 
    if inconsistent:
        analysis_description += f"Inconsistency found: {inconsistency_reason}<br><b>Conclusion: No solution exists.</b>"
        final_solution_type = "inconsistent"
        final_solution_desc = "No solution exists."
    else:
        analysis_description += "No inconsistencies found (no rows of the form [0 0 ... 0 | c] where c ≠ 0).<br>Proceeding to analyze solution type based on rank and number of variables."
        final_solution_type = None # Determine below
        final_solution_desc = None # Determine below
        
    rref_steps.append({
        'description': analysis_description,
        'matrix': format_matrix(A_rref),
        'pivot_coord': None, # No specific pivot for this analysis step
        'all_pivot_coords': pivots
    })
    current_step_num += 1

    # --- Step 3: Determine Solution Type and Describe (if consistent) --- 
    if not inconsistent:
        solution_analysis_desc = f"<b>Step {current_step_num}: Analyze Solution Type</b><br>"
        solution_analysis_desc += f"Rank of the augmented matrix (number of pivots) = {rank}.<br>"
        solution_analysis_desc += f"Number of variables = {num_vars}.<br>"
        
        if rank == num_vars:
            # Unique solution
            final_solution_type = "unique"
            solution_analysis_desc += "Rank equals the number of variables.<br><b>Conclusion: Unique solution exists.</b><br>"
            solution_values = ["0"] * num_vars # Initialize solutions
            for r, c in pivots:
                 if c < num_vars: # Ensure pivot corresponds to a variable
                     solution_values[c] = f"{A_rref[r, -1]:.4f}" # Value from the constant column
            
            final_solution_desc = "Unique solution: " + ", ".join([f"x<sub>{i+1}</sub> = {val}" for i, val in enumerate(solution_values)])
            solution_analysis_desc += final_solution_desc
            
        else: # rank < num_vars
            # Infinite solutions
            final_solution_type = "infinite"
            free_var_indices = [c for c in range(num_vars) if c not in pivot_cols]
            free_var_names = [f"x<sub>{idx+1}</sub>" for idx in free_var_indices]
            
            solution_analysis_desc += "Rank is less than the number of variables.<br><b>Conclusion: Infinitely many solutions exist.</b><br>"
            solution_analysis_desc += f"Free variables: {', '.join(free_var_names)}.<br>"
            
            # Express basic variables in terms of free variables
            basic_var_expr = []
            for r, c in pivots: # c is the basic variable index
                if c < num_vars:
                    expr = f"x<sub>{c+1}</sub> = {A_rref[r, -1]:.4f}"
                    for free_idx in free_var_indices:
                        coeff = A_rref[r, free_idx]
                        if abs(coeff) > 1e-10:
                            expr += f" {-coeff:.4f} x<sub>{free_idx+1}</sub>"
                            # Basic formatting for sign
                            expr = expr.replace("+ -", "- ").replace("- -", "+ ")
                    basic_var_expr.append(expr)
            
            final_solution_desc = "Solution parameterization:<br>" + "<br>".join(basic_var_expr)
            # Add free variables explicitly
            for free_var_name in free_var_names:
                 final_solution_desc += f"<br>{free_var_name} = {free_var_name} (free)"
                 
            solution_analysis_desc += final_solution_desc
            
        rref_steps.append({
            'description': solution_analysis_desc,
            'matrix': format_matrix(A_rref),
            'pivot_coord': None,
            'all_pivot_coords': pivots
        })
        current_step_num += 1 # Increment even if no solution found
        
    # Add a final summary step for clarity
    summary_desc = f"<b>Step {current_step_num}: Solution Summary</b><br>"
    summary_desc += f"Solution Type: <b>{final_solution_type.upper()}</b><br>"
    summary_desc += f"{final_solution_desc}"
    
    rref_steps.append({
         'description': summary_desc,
         'matrix': format_matrix(A_rref), # Show final matrix again
         'pivot_coord': None,
         'all_pivot_coords': pivots
     })

    return rref_steps

# Test with example matrices
if __name__ == "__main__":
    # Non-homogeneous system (unique solution)
    print("Test 1: Non-homogeneous system (unique solution)")
    m1 = [
        [1, 2, -1, 7,-1],
        [3, 8, 2, 8,28],
        [4, 9, -1, 9,14],
        [1, 2, -1, 9,14]
    ]
    print("Original Augmented Matrix:")
    print(np.array(m1))
    print()
    result1 = gauss_jordan_solver(m1)
    print("Result:", result1)
    print()

    # Homogeneous system (infinite solutions)
    # print("Test 2: Homogeneous system (infinite solutions)")
    # m2 = [
    #     [1, 2, 0, 0],
    #     [0, -1, 2, 0],
    #     [0, 0, 0, 0]
    # ]
    # print("Original Augmented Matrix:")
    # print(np.array(m2))
    # print()
    # result2 = gauss_jordan_solver(m2)
    # print("Result:", result2)
    # print()

    # # Inconsistent system
    # print("Test 3: Inconsistent system")
    # m3 = [
    #     [1, 2, 3, 4],
    #     [2, 4, 6, 7]
    # ]
    # print("Original Augmented Matrix:")
    # print(np.array(m3))
    # print()
    # result3 = gauss_jordan_solver(m3)
    # print("Result:", result3)
    # print()

    # Test function using the new steps function
    print("--- Test 1: Unique Solution ---")
    m1 = [
        [1, 1, 2, 9],
        [2, 4, -3, 1],
        [3, 6, -5, 0]
    ]
    print("Original Augmented Matrix:")
    print(np.array(m1))
    
    try:
        eq_steps = solve_equations_steps(m1)
        print(f"\nTotal steps: {len(eq_steps)}")
        # Print only the last few steps (analysis)
        print("\n--- ANALYSIS STEPS ---")
        start_index = max(0, len(eq_steps) - 3) # Show last 3 steps
        for i in range(start_index, len(eq_steps)):
             step = eq_steps[i]
             print(f"\nStep {i+1}: {step['description']}")
             print("Matrix:")
             print(np.array(step['matrix']))
             # print(f"Pivots: {step['all_pivot_coords']}")
    except ValueError as e:
        print(f"Error: {e}")
        
    print("\n--- Test 2: Infinite Solutions ---")
    m2 = [
        [1, -2, 1, 0],
        [2, 1, -3, 5],
        [4, -7, 1, -1]
    ]
    print("Original Augmented Matrix:")
    print(np.array(m2))
    try:
        eq_steps_2 = solve_equations_steps(m2)
        print(f"\nTotal steps: {len(eq_steps_2)}")
        print("\n--- ANALYSIS STEPS --- ")
        start_index = max(0, len(eq_steps_2) - 3) # Show last 3 steps
        for i in range(start_index, len(eq_steps_2)):
             step = eq_steps_2[i]
             print(f"\nStep {i+1}: {step['description']}")
             print("Matrix:")
             print(np.array(step['matrix']))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test 3: Inconsistent System ---")
    m3 = [
        [1, 1, -1, 2],
        [2, 2, -1, 5],
        [1, 1, 0, 4]
    ]
    print("Original Augmented Matrix:")
    print(np.array(m3))
    try:
        eq_steps_3 = solve_equations_steps(m3)
        print(f"\nTotal steps: {len(eq_steps_3)}")
        print("\n--- ANALYSIS STEPS --- ")
        start_index = max(0, len(eq_steps_3) - 3) # Show last 3 steps
        for i in range(start_index, len(eq_steps_3)):
             step = eq_steps_3[i]
             print(f"\nStep {i+1}: {step['description']}")
             print("Matrix:")
             print(np.array(step['matrix']))
    except ValueError as e:
        print(f"Error: {e}")
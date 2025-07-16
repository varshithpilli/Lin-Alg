import numpy as np
from math import gcd
from . import rref
from . import utils

# def gauss_jordan_solver(augmented_matrix):
#     """
#     Solve a system of linear equations (homogeneous or non-homogeneous) using Gauss-Jordan elimination.
#     Computes the RREF of the augmented matrix and extracts solutions for the variables.
#     Uses integer-preserving operations with minimized m, n (divided by GCD) and positive m.
#     Returns a dictionary with RREF, solution type, and variable values or solution space description.
#     If verbose=True, prints step-by-step operations using 1-based indexing.
#     """
#     A = np.array(augmented_matrix, dtype=float)
#     rows, cols = A.shape
#     num_vars = cols - 1

#     r = 0
#     step = 0
#     pivot_columns = []

#     while r < rows:
#         pivot_col = -1
#         for j in range(cols - 1):
#             if abs(A[r, j]) > 1e-10 and j not in pivot_columns:
#                 pivot_col = j
#                 break

#         if pivot_col == -1:
#             if abs(A[r, cols-1]) > 1e-10:
#                 return {
#                     "rref": A.tolist(),
#                     "solution_type": "inconsistent",
#                     "solution": "No solution"
#                 }
#             step += 1
#             r += 1
#             continue

#         pivot_columns.append(pivot_col)
#         pivot_val = A[r, pivot_col]

#         if abs(pivot_val - 1.0) > 1e-10:
#             step += 1
#             A[r] = A[r] / pivot_val
            
#         for i in range(rows):
#             if i != r and abs(A[i, pivot_col]) > 1e-10:
#                 m = A[r, pivot_col]
#                 n = A[i, pivot_col]
#                 m_int = int(round(m))
#                 n_int = int(round(n))
#                 g = gcd(abs(m_int), abs(n_int)) if m_int != 0 and n_int != 0 else 1
#                 if g > 1:
#                     m = m / g
#                     n = n / g
#                 if m < 0:
#                     m = -m
#                     n = -n
#                 step += 1
#                 A[i] = m * A[i] - n * A[r]
#                 A[i, pivot_col] = 0.0
                
#         r += 1

#     if np.all(np.abs(A - np.round(A)) < 1e-10):
#         A = np.round(A).astype(int)

#     rank = len(pivot_columns)
#     is_homogeneous = np.all(np.abs(A[:, -1]) < 1e-10)

    
#     for i in range(rows):
#         if all(abs(A[i, j]) < 1e-10 for j in range(cols-1)) and abs(A[i, cols-1]) > 1e-10:
#             return {
#                 "rref": A.tolist(),
#                 "solution_type": "inconsistent",
#                 "solution": "No solution"
#             }

#     if rank == num_vars and rank <= rows:
#         solution = [A[i, cols-1] for i in range(min(rows, num_vars)) if i < len(pivot_columns)]
#         solution += [0.0] * (num_vars - len(solution))  
#         solution_dict = {f"x{i+1}": val for i, val in enumerate(solution)}
#         return {
#             "rref": A.tolist(),
#             "solution_type": "unique",
#             "solution": solution_dict
#         }
#     else:
#         free_vars = [i for i in range(num_vars) if i not in pivot_columns]
#         solution_desc = "Infinite solutions with free variables: "
#         solution_desc += ", ".join([f"x{i+1}" for i in free_vars]) if free_vars else "None (trivial solution)"
        
#         if is_homogeneous:
#             if free_vars:
#                 solution_desc += "\nNon-trivial solutions exist. General solution involves linear combinations of free variables."
#             else:
#                 solution_desc = "Trivial solution: " + ", ".join([f"x{i+1}=0" for i in range(num_vars)])
#         else:
#             particular = [0.0] * num_vars
#             for i, col in enumerate(pivot_columns):
#                 if i < rows:
#                     particular[col] = A[i, cols-1]
#             solution_desc += f"\nParticular solution: {', '.join([f'x{i+1}={val}' for i, val in enumerate(particular)])}"
#             if free_vars:
#                 solution_desc += "\nGeneral solution: Particular solution + linear combinations of free variables."

#         return {
#             "rref": A.tolist(),
#             "solution_type": "infinite",
#             "solution": solution_desc
#         }

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

    rref_steps = rref.row_reduced_echelon_form_steps(augmented_matrix)
    
    final_rref_step = rref_steps[-1]
    A_rref = np.array(final_rref_step['matrix'], dtype=float)
    rows, cols = A_rref.shape
    num_vars = cols - 1
    pivots = final_rref_step['all_pivot_coords']
    pivot_cols = sorted([p[1] for p in pivots])
    rank = len(pivot_cols)
    current_step_num = len(rref_steps) + 1

    inconsistent = False
    inconsistency_reason = ""
    for r in range(rows):
        row_is_zero_coeffs = all(abs(A_rref[r, c]) < 1e-10 for c in range(num_vars))
        if row_is_zero_coeffs and abs(A_rref[r, -1]) > 1e-10:
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
        final_solution_type = None
        final_solution_desc = None
        
    rref_steps.append({
        'description': analysis_description,
        'matrix': utils.format_matrix(A_rref),
        'pivot_coord': None,
        'all_pivot_coords': pivots
    })
    current_step_num += 1

    if not inconsistent:
        solution_analysis_desc = f"<b>Step {current_step_num}: Analyze Solution Type</b><br>"
        solution_analysis_desc += f"Rank of the augmented matrix (number of pivots) = {rank}.<br>"
        solution_analysis_desc += f"Number of variables = {num_vars}.<br>"
        
        if rank == num_vars:
            final_solution_type = "unique"
            solution_analysis_desc += "Rank equals the number of variables.<br><b>Conclusion: Unique solution exists.</b><br>"
            solution_values = ["0"] * num_vars
            for r, c in pivots:
                 if c < num_vars:
                     solution_values[c] = f"{A_rref[r, -1]:.4f}"
            
            final_solution_desc = "Unique solution: " + ", ".join([f"x<sub>{i+1}</sub> = {val}" for i, val in enumerate(solution_values)])
            solution_analysis_desc += final_solution_desc
            
        else:
            final_solution_type = "infinite"
            free_var_indices = [c for c in range(num_vars) if c not in pivot_cols]
            free_var_names = [f"x<sub>{idx+1}</sub>" for idx in free_var_indices]
            
            solution_analysis_desc += "Rank is less than the number of variables.<br><b>Conclusion: Infinitely many solutions exist.</b><br>"
            solution_analysis_desc += f"Free variables: {', '.join(free_var_names)}.<br>"
            
            basic_var_expr = []
            for r, c in pivots:
                if c < num_vars:
                    expr = f"x<sub>{c+1}</sub> = {A_rref[r, -1]:.4f}"
                    for free_idx in free_var_indices:
                        coeff = A_rref[r, free_idx]
                        if abs(coeff) > 1e-10:
                            expr += f" {-coeff:.4f} x<sub>{free_idx+1}</sub>"
                            expr = expr.replace("+ -", "- ").replace("- -", "+ ")
                    basic_var_expr.append(expr)
            
            final_solution_desc = "Solution parameterization:<br>" + "<br>".join(basic_var_expr)
            for free_var_name in free_var_names:
                 final_solution_desc += f"<br>{free_var_name} = {free_var_name} (free)"
                 
            solution_analysis_desc += final_solution_desc
            
        rref_steps.append({
            'description': solution_analysis_desc,
            'matrix': utils.format_matrix(A_rref),
            'pivot_coord': None,
            'all_pivot_coords': pivots
        })
        current_step_num += 1

    summary_desc = f"<b>Step {current_step_num}: Solution Summary</b><br>"
    summary_desc += f"Solution Type: <b>{final_solution_type.upper()}</b><br>"
    summary_desc += f"{final_solution_desc}"
    
    rref_steps.append({
         'description': summary_desc,
         'matrix': utils.format_matrix(A_rref),
         'pivot_coord': None,
         'all_pivot_coords': pivots
     })

    return rref_steps

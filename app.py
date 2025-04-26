import numpy as np
from flask import Flask, render_template, request, jsonify
# Import calculation functions from separate files
from inverse import inverse_gauss_jordan
# Import eigen function (assuming it's in eigen.py, adjust if needed)
# from eigen import eigen # We need to check if eigen.py exists
# Import gram_schmidt (assuming ortho.py)
# from ortho import gram_schmidt # We need to check if ortho.py exists
# Import matrix_space_bases from basis.py
from basis import matrix_space_bases
# Import REF function from ref.py
from ref import row_echelon_form_steps
# Import RREF function from rref.py
from rref import row_reduced_echelon_form_steps
# Import Equation Solver function from eqns.py
from eqns import solve_equations_steps

app = Flask(__name__)

def format_vector(v, decimals=5):
    """Helper to format numpy vector for display, rounding near-zero."""
    # Round each element to the specified number of decimals, return as float
    return [round(float(x), decimals) if abs(x) > 1e-10 else 0.0 for x in v]

def format_scalar(s, decimals=5):
    """Helper to format scalar for display, rounding near-zero."""
    # Round the scalar to the specified number of decimals, return as float
    return round(float(s), decimals) if abs(s) > 1e-10 else 0.0

def gram_schmidt(vectors):
    if not vectors:
        raise ValueError("No vectors provided.")
    
    try:
        vectors_np = [np.array(v, dtype=float) for v in vectors]
        # Check dimensions early
        if not vectors_np:
            raise ValueError("Input vector list is empty after conversion.")
        n = len(vectors_np[0])
        if n == 0:
            raise ValueError("Vectors cannot have zero dimension.")
        if not all(len(v) == n for v in vectors_np):
            raise ValueError("All vectors must have the same dimension.")
    except (ValueError, IndexError, TypeError) as e: # Catch TypeError for non-numeric elements
        raise ValueError(f"Invalid input vectors: {e}. Ensure they are non-empty lists of numbers and have the same dimension.")
    
    num_vectors = len(vectors_np)
    steps = [] # Initialize steps list
    orthogonal_basis_u = [] # Store intermediate orthogonal vectors (u)
    orthonormal_basis_e = [] # Store final orthonormal vectors (e)

    # Initial step: Show the input vectors
    steps.append({
        "description": f"<b>Starting with the initial set of {num_vectors} vector(s) (v):</b>",
        "vectors": [format_vector(v) for v in vectors_np],
        "vector_indices": list(range(1, num_vectors + 1)), # Indices 1 to N
        "vector_type": "v"
    })
    
    for i, v_i in enumerate(vectors_np):
        u_i = v_i.copy()
        
        # Step 1: Initialize u_i = v_i
        steps.append({
            "description": (
                f"<b>Step {len(steps)}: Initialize u<sub>{i+1}</sub></b><br>"
                f"<span class='formula'>u<sub>{i+1}</sub> = v<sub>{i+1}</sub> = {format_vector(v_i)}</span>"
            ),
            "vectors": [format_vector(u_i)], 
            "vector_indices": [i + 1], # Index for this specific u_i
            "vector_type": "u"
        })
        
        # Step 2: Subtract projections onto previous orthonormal vectors e_j
        step_description_projections = (
            f"<b>Step {len(steps)}: Subtract projections from u<sub>{i+1}</sub></b><br>"
            f"<span class='formula'>u<sub>{i+1}</sub> = u<sub>{i+1}</sub> - Σ<sub>j=1</sub><sup>{len(orthonormal_basis_e)}</sup> proj<sub>e<sub>j</sub></sub>(v<sub>{i+1}</sub>)</span>"
        )
        if not orthonormal_basis_e:
            step_description_projections += " (No previous e vectors to project onto)"
        else:
            step_description_projections += ":"
            
        u_intermediate = u_i.copy() # Keep track for description
        projection_details = []

        for j, e_j in enumerate(orthonormal_basis_e):
            proj_val = np.dot(v_i, e_j) # Calculate dot product v_i ⋅ e_j
            projection_vector = proj_val * e_j # Calculate projection vector
            
            # Format parts of the formula description
            formula_part1 = f"<span class='formula'>v<sub>{i+1}</sub> ⋅ e<sub>{j+1}</sub> = {format_vector(v_i)} ⋅ {format_vector(e_j)} = {format_scalar(proj_val)}</span>"
            formula_part2 = f"<span class='formula'>proj<sub>e<sub>{j+1}</sub></sub>(v<sub>{i+1}</sub>) = (v<sub>{i+1}</sub> ⋅ e<sub>{j+1}</sub>) * e<sub>{j+1}</sub> = {format_scalar(proj_val)} * {format_vector(e_j)} = {format_vector(projection_vector)}</span>"
            formula_part3 = f"<span class='formula'>u<sub>{i+1}</sub> = u<sub>{i+1}</sub> - proj = {format_vector(u_intermediate)} - {format_vector(projection_vector)}</span>"
            
            u_i -= projection_vector # Update u_i
            u_intermediate = u_i.copy() # Update intermediate for next description line
            formula_part4 = f"<span class='formula'> = {format_vector(u_i)}</span>"
            
            detail = (
                f"<div style='margin-left: 20px; margin-top: 5px;'>"
                f"For j={j+1}:<br>"
                f"{formula_part1}<br>"
                f"{formula_part2}<br>"
                f"{formula_part3}{formula_part4}"
                f"</div>"
            )
            projection_details.append(detail)
        
        if projection_details:
            step_description_projections += "<br>" + "<br>".join(projection_details)
        else:
             step_description_projections += "<br>&nbsp;&nbsp;(No projections to subtract as j starts from 1)" # Handle case i=0
            
        # Add the projection step results
        steps.append({
            "description": step_description_projections,
            "vectors": [format_vector(u_i)], # Show resulting u_i after projections
            "vector_indices": [i + 1], # Index for this specific u_i
            "vector_type": "u"
        })

        # Step 3: Check norm and Normalize u_i to get e_i
        norm_ui = np.linalg.norm(u_i)
        
        if norm_ui < 1e-10:
            # Vector is linearly dependent
            steps.append({
                "description": (
                    f"<b>Step {len(steps)}: Check Norm</b><br>"
                    f"<span class='formula'>||u<sub>{i+1}</sub>|| = ||{format_vector(u_i)}|| ≈ {format_scalar(norm_ui, decimals=8)}</span><br>"
                    f"Since the norm is close to zero, v<sub>{i+1}</sub> is linearly dependent on the previous vectors. Skipping normalization."
                ),
                "vectors": [], # No new vector added
                "vector_indices": [], # No index needed
                "vector_type": "info" 
            })
            continue # Skip to the next v_i
        
        # Normalize u_i to get e_i
        e_i = u_i / norm_ui
        orthonormal_basis_e.append(e_i)
        orthogonal_basis_u.append(u_i) 
        
        steps.append({
            "description": (
                f"<b>Step {len(steps)}: Normalize u<sub>{i+1}</sub> to get e<sub>{i+1}</sub></b><br>"
                f"<div style='margin-left: 20px; margin-top: 5px;'>"
                f"<span class='formula'>||u<sub>{i+1}</sub>|| = ||{format_vector(u_i)}|| = {format_scalar(norm_ui)}</span><br>"
                f"<span class='formula'>e<sub>{i+1}</sub> = u<sub>{i+1}</sub> / ||u<sub>{i+1}</sub>|| = {format_vector(u_i)} / {format_scalar(norm_ui)}</span><br>"
                f"<span class='formula'>Resulting e<sub>{i+1}</sub> = {format_vector(e_i)}</span>"
                f"</div>"
            ),
            "vectors": [format_vector(e_i)], # Show only the calculated e_i
            "vector_indices": [i + 1], # Index for this specific e_i
            "vector_type": "e"
        })

    # Final step: Show the complete orthonormal basis
    if orthonormal_basis_e:
         steps.append({
            "description": f"<b>Step {len(steps)}: Final Orthonormal Basis (e)</b>",
            "vectors": [format_vector(e_vec) for e_vec in orthonormal_basis_e],
            "vector_indices": list(range(1, len(orthonormal_basis_e) + 1)), # Indices 1 to num_basis_vectors
            "vector_type": "e"
        })
    else:
         steps.append({
            "description": f"<b>Step {len(steps)}: Final Result</b><br>No linearly independent vectors found. The orthonormal basis is empty.",
            "vectors": [],
            "vector_indices": [], # No index needed
            "vector_type": "info"
        })
    
    return steps

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate/inverse', methods=['POST'])
def calculate_inverse():
    data = request.get_json()
    try:
        matrix = np.array(data['matrix'], dtype=float)
        steps = inverse_gauss_jordan(matrix)
        return jsonify({
            'success': True,
            'steps': steps
        })
    except Exception as e:
        print(f"Error during inverse calculation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/calculate/orthonormal', methods=['POST'])
def calculate_orthonormal():
    data = request.get_json()
    try:
        # Input validation happens inside gram_schmidt now
        vectors = data.get('vectors', []) 
        if not isinstance(vectors, list) or not all(isinstance(v, list) for v in vectors):
             raise ValueError("Invalid input format. 'vectors' must be a list of lists.")

        steps = gram_schmidt(vectors)
        return jsonify({
            # 'success': True, # Not strictly needed if using steps/error structure
            'steps': steps
            # Removed 'orthonormal_basis' and 'message' as they are now part of steps
        })
    except Exception as e:
        print(f"Error during orthonormal calculation: {e}") # Log the error server-side
        import traceback
        traceback.print_exc()
        return jsonify({
            # 'success': False,
            'error': str(e) # Send error message to frontend
        }), 400 # Return a 400 Bad Request status code for client errors

@app.route('/calculate/bases', methods=['POST'])
def calculate_bases():
    data = request.get_json()
    try:
        matrix = np.array(data['matrix'], dtype=float)
        # Call the modified function which now returns only steps
        steps = matrix_space_bases(matrix) # Correctly assign the single returned list
        return jsonify({
            # 'success': True, # Not strictly needed
            'steps': steps,
            # 'final_results': final_results # Removed
        })
    except ValueError as ve: # Catch specific ValueErrors from basis.py validation
        print(f"Validation Error during basis calculation: {ve}")
        return jsonify({
            # 'success': False,
            'error': str(ve) # Return the specific validation error message
        }), 400 # Bad Request status code
    except Exception as e:
        print(f"Error during basis calculation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            # 'success': False,
            'error': f"An unexpected server error occurred: {e}" # Generic error for other exceptions
        }), 500 # Internal Server Error status code

@app.route('/calculate/ref', methods=['POST'])
def calculate_ref():
    data = request.get_json()
    try:
        matrix = data.get('matrix')
        if matrix is None:
             raise ValueError("Missing 'matrix' in request data.")
             
        # The validation is now inside row_echelon_form_steps
        steps = row_echelon_form_steps(matrix)
        return jsonify({
            'steps': steps
        })
    except ValueError as ve: # Catch specific ValueErrors from ref.py validation
        print(f"Validation Error during REF calculation: {ve}")
        return jsonify({'error': str(ve)}), 400 # Bad Request
    except Exception as e:
        print(f"Error during REF calculation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An unexpected server error occurred: {e}"}), 500 # Internal Server Error

@app.route('/calculate/rref', methods=['POST'])
def calculate_rref():
    data = request.get_json()
    try:
        matrix = data.get('matrix')
        if matrix is None:
             raise ValueError("Missing 'matrix' in request data.")
             
        # Validation is inside row_reduced_echelon_form_steps
        steps = row_reduced_echelon_form_steps(matrix)
        return jsonify({
            'steps': steps
        })
    except ValueError as ve:
        print(f"Validation Error during RREF calculation: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error during RREF calculation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An unexpected server error occurred: {e}"}), 500

@app.route('/calculate/equations', methods=['POST'])
def calculate_equations():
    data = request.get_json()
    try:
        # Input is the augmented matrix
        matrix = data.get('matrix') 
        if matrix is None:
             raise ValueError("Missing 'matrix' (augmented) in request data.")
             
        # Function performs RREF and solution analysis, returns all steps
        steps = solve_equations_steps(matrix)
        return jsonify({
            'steps': steps
        })
    except ValueError as ve: # Catch validation errors from eqns.py
        print(f"Validation Error during Equation Solving: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error during Equation Solving: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An unexpected server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True) 
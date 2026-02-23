import numpy as np
from flask import Flask, request, jsonify
from logic import basis, eqns, inverse, ortho, rref, ref
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://varzone.in"}})
# CORS(app, resources={r"/*": {"origins": "*"}})

# landing page
@app.route('/')
def index():
    return "Nothing to see here"

# matrix inverse
@app.route('/calculate/inverse', methods=['POST'])
def calculate_inverse():
    data = request.get_json()
    try:
        matrix = np.array(data['matrix'], dtype=float)
        steps = inverse.inverse_gauss_jordan(matrix)
        return jsonify({
            'success': True,
            'steps': steps
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# gram-schmidt
@app.route('/calculate/orthonormal', methods=['POST'])
def calculate_orthonormal():
    data = request.get_json()
    try:
        vectors = data.get('vectors', []) 
        if not isinstance(vectors, list) or not all(isinstance(v, list) for v in vectors):
             raise ValueError("Invalid input format. 'vectors' must be a list of lists.")

        steps = ortho.gram_schmidt(vectors)
        return jsonify({
            'success': True,
            'steps': steps
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
# rank-nullity
@app.route('/calculate/bases', methods=['POST'])
def calculate_bases():
    data = request.get_json()
    try:
        matrix = np.array(data['matrix'], dtype=float)
        steps = basis.matrix_space_bases(matrix)
        return jsonify({
            'success': True,
            'steps': steps,
        })
    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"An unexpected server error occurred: {e}"
        }), 500

# REF
@app.route('/calculate/ref', methods=['POST'])
def calculate_ref():
    data = request.get_json()
    try:
        matrix = data.get('matrix')
        if matrix is None:
             raise ValueError("Missing 'matrix' in request data.")
             
        steps = ref.row_echelon_form_steps(matrix)
        return jsonify({
            'success': True,
            'steps': steps
        })
    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"An unexpected server error occurred: {e}"
        }),

# RREF
@app.route('/calculate/rref', methods=['POST'])
def calculate_rref():
    data = request.get_json()
    try:
        matrix = data.get('matrix')
        if matrix is None:
             raise ValueError("Missing 'matrix' in request data.")
             
        steps = rref.row_reduced_echelon_form_steps(matrix)
        return jsonify({
            'success': True,
            'steps': steps
        })
    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"An unexpected server error occurred: {e}"
        }), 500

# solve eqns
@app.route('/calculate/equations', methods=['POST'])
def calculate_equations():
    data = request.get_json()
    try:
        matrix = data.get('matrix') 
        if matrix is None:
             raise ValueError("Missing 'matrix' (augmented) in request data.")
             
        steps = eqns.solve_equations_steps(matrix)
        return jsonify({
            'success': True,
            'steps': steps
        })
    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"An unexpected server error occurred: {e}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 
import numpy as np

def format_vector(v, decimals=5):
    """Helper to format numpy vector for display, rounding near-zero."""
    return [round(float(x), decimals) if abs(x) > 1e-10 else 0.0 for x in v]

def format_scalar(s, decimals=5):
    """Helper to format scalar for display, rounding near-zero."""
    return round(float(s), decimals) if abs(s) > 1e-10 else 0.0

def format_matrix(matrix):
    """Helper to format numpy matrix for JSON serialization, handling potential near-zero floats."""
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    return [[float(0.0) if abs(x) < 1e-10 else float(x) for x in row] for row in matrix]
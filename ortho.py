import numpy as np

def gram_schmidt(vectors):
    if not vectors:
        raise ValueError("No vectors provided.")
    vectors = [np.array(v, dtype=float) for v in vectors]
    n = len(vectors[0])
    if not all(len(v) == n for v in vectors):
        raise ValueError("All vectors must have the same dimension.")
    
    orthonormal_basis = []
    message_parts = []
    
    for i, v in enumerate(vectors):
        u = v.copy()
        
        for e in orthonormal_basis:
            projection = np.dot(v, e) * e
            u -= projection
        
        norm = np.linalg.norm(u)
        
        if norm < 1e-10:
            message_parts.append(f"Vector {i+1} is linearly dependent and was skipped.")
            continue
        
        e = u / norm
        orthonormal_basis.append(e)
        message_parts.append(f"Vector {i+1} processed successfully.")
    
    if not orthonormal_basis:
        message = "No linearly independent vectors found."
    else:
        message = "Gram-Schmidt orthonormalization completed:\n" + "\n".join(message_parts)
    
    return {
        'orthonormal_basis': orthonormal_basis,
        'message': message
    }

if __name__ == "__main__":
    v1 = np.array([1, 1, 0, 1])
    v2 = np.array([0, 1, 1, 1])
    v3 = np.array([-1, 0, 0, 1])
    result1 = gram_schmidt([v1, v2, v3])
    print("Test 1: Three independent vectors in R^3")
    print(result1['message'])
    print("Orthonormal basis:")
    for e in result1['orthonormal_basis']:
        print(e.tolist())
    print()
    
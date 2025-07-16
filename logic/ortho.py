import numpy as np
import utils

def gram_schmidt(vectors):
    if not vectors:
        raise ValueError("No vectors provided.")
    
    try:
        vectors_np = [np.array(v, dtype=float) for v in vectors]
        if not vectors_np:
            raise ValueError("Input vector list is empty after conversion.")
        n = len(vectors_np[0])
        if n == 0:
            raise ValueError("Vectors cannot have zero dimension.")
        if not all(len(v) == n for v in vectors_np):
            raise ValueError("All vectors must have the same dimension.")
    except (ValueError, IndexError, TypeError) as e:
        raise ValueError(f"Invalid input vectors: {e}. Ensure they are non-empty lists of numbers and have the same dimension.")
    
    num_vectors = len(vectors_np)
    steps = [] 
    orthogonal_basis_u = []
    orthonormal_basis_e = [] 

    steps.append({
        "description": f"<b>Starting with the initial set of {num_vectors} vector(s) (v):</b>",
        "vectors": [utils.format_vector(v) for v in vectors_np],
        "vector_indices": list(range(1, num_vectors + 1)),
        "vector_type": "v"
    })
    
    for i, v_i in enumerate(vectors_np):
        u_i = v_i.copy()
        
        steps.append({
            "description": (
                f"<b>Step {len(steps)}: Initialize u<sub>{i+1}</sub></b><br>"
                f"<span class='formula'>u<sub>{i+1}</sub> = v<sub>{i+1}</sub> = {utils.format_vector(v_i)}</span>"
            ),
            "vectors": [utils.format_vector(u_i)], 
            "vector_indices": [i + 1], 
            "vector_type": "u"
        })
        
        step_description_projections = (
            f"<b>Step {len(steps)}: Subtract projections from u<sub>{i+1}</sub></b><br>"
            f"<span class='formula'>u<sub>{i+1}</sub> = u<sub>{i+1}</sub> - Σ<sub>j=1</sub><sup>{len(orthonormal_basis_e)}</sup> proj<sub>e<sub>j</sub></sub>(v<sub>{i+1}</sub>)</span>"
        )
        if not orthonormal_basis_e:
            step_description_projections += " (No previous e vectors to project onto)"
        else:
            step_description_projections += ":"
            
        u_intermediate = u_i.copy()
        projection_details = []

        for j, e_j in enumerate(orthonormal_basis_e):
            proj_val = np.dot(v_i, e_j)
            projection_vector = proj_val * e_j 
            
            formula_part1 = f"<span class='formula'>v<sub>{i+1}</sub> ⋅ e<sub>{j+1}</sub> = {utils.format_vector(v_i)} ⋅ {utils.format_vector(e_j)} = {utils.format_scalar(proj_val)}</span>"
            formula_part2 = f"<span class='formula'>proj<sub>e<sub>{j+1}</sub></sub>(v<sub>{i+1}</sub>) = (v<sub>{i+1}</sub> ⋅ e<sub>{j+1}</sub>) * e<sub>{j+1}</sub> = {utils.format_scalar(proj_val)} * {utils.format_vector(e_j)} = {utils.format_vector(projection_vector)}</span>"
            formula_part3 = f"<span class='formula'>u<sub>{i+1}</sub> = u<sub>{i+1}</sub> - proj = {utils.format_vector(u_intermediate)} - {utils.format_vector(projection_vector)}</span>"
            
            u_i -= projection_vector
            u_intermediate = u_i.copy()
            formula_part4 = f"<span class='formula'> = {utils.format_vector(u_i)}</span>"
            
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
             step_description_projections += "<br>&nbsp;&nbsp;(No projections to subtract as j starts from 1)"
            
        steps.append({
            "description": step_description_projections,
            "vectors": [utils.format_vector(u_i)],
            "vector_indices": [i + 1],
            "vector_type": "u"
        })

        norm_ui = np.linalg.norm(u_i)
        
        if norm_ui < 1e-10:
            steps.append({
                "description": (
                    f"<b>Step {len(steps)}: Check Norm</b><br>"
                    f"<span class='formula'>||u<sub>{i+1}</sub>|| = ||{utils.format_vector(u_i)}|| ≈ {utils.format_scalar(norm_ui, decimals=8)}</span><br>"
                    f"Since the norm is close to zero, v<sub>{i+1}</sub> is linearly dependent on the previous vectors. Skipping normalization."
                ),
                "vectors": [],
                "vector_indices": [],
                "vector_type": "info" 
            })
            continue 
        
        e_i = u_i / norm_ui
        orthonormal_basis_e.append(e_i)
        orthogonal_basis_u.append(u_i) 
        
        steps.append({
            "description": (
                f"<b>Step {len(steps)}: Normalize u<sub>{i+1}</sub> to get e<sub>{i+1}</sub></b><br>"
                f"<div style='margin-left: 20px; margin-top: 5px;'>"
                f"<span class='formula'>||u<sub>{i+1}</sub>|| = ||{utils.format_vector(u_i)}|| = {utils.format_scalar(norm_ui)}</span><br>"
                f"<span class='formula'>e<sub>{i+1}</sub> = u<sub>{i+1}</sub> / ||u<sub>{i+1}</sub>|| = {utils.format_vector(u_i)} / {utils.format_scalar(norm_ui)}</span><br>"
                f"<span class='formula'>Resulting e<sub>{i+1}</sub> = {utils.format_vector(e_i)}</span>"
                f"</div>"
            ),
            "vectors": [utils.format_vector(e_i)],
            "vector_indices": [i + 1],
            "vector_type": "e"
        })

    if orthonormal_basis_e:
         steps.append({
            "description": f"<b>Step {len(steps)}: Final Orthonormal Basis (e)</b>",
            "vectors": [utils.format_vector(e_vec) for e_vec in orthonormal_basis_e],
            "vector_indices": list(range(1, len(orthonormal_basis_e) + 1)),
            "vector_type": "e"
        })
    else:
         steps.append({
            "description": f"<b>Step {len(steps)}: Final Result</b><br>No linearly independent vectors found. The orthonormal basis is empty.",
            "vectors": [],
            "vector_indices": [], 
            "vector_type": "info"
        })
    
    return steps
import numpy as np

def format_matrix(mat):
    if not isinstance(mat, np.ndarray):
        try:
            mat = np.array(mat, dtype=float) 
        except ValueError:
            return []
        except TypeError:
             return []

    if mat.size == 0:
         return []

    if np.iscomplexobj(mat):
        return [[{'real': round(val.real, 4), 'imag': round(val.imag, 4)} for val in row] for row in mat.tolist()]
    else:
        return [[round(float(val), 4) for val in row] for row in mat.tolist()]

def format_vectors_for_desc(basis_vectors, prefix):
    if not basis_vectors:
        return "&nbsp;&nbsp;{(Empty Set)}<br>"
    desc = ""
    for i, vec in enumerate(basis_vectors):
        vec_str = ", ".join(map(str, vec))
        desc += f"&nbsp;&nbsp;{prefix}<sub>{i+1}</sub>: [{vec_str}]<br>"
    return desc

def matrix_space_bases(A):
    A_orig = np.array(A, dtype=float) 
    if A_orig.size == 0:
        raise ValueError("Input matrix is empty.")
    
    m, n = A_orig.shape
    A_copy = A_orig.copy()
    steps = []
    pivot_indices = []
    rank = 0
    original_pivot_row_indices = []
    row_indices_tracker = list(range(m))
    current_step_num = 1

    steps.append({
        'description': f'<b>Step {current_step_num}: Start with the Original Matrix A</b>',
        'matrix': format_matrix(A_copy),
        'all_pivot_coords': []
    })
    current_step_num += 1

    col = 0
    while rank < m and col < n:
        pivot_row_idx = rank
        while pivot_row_idx < m and abs(A_copy[pivot_row_idx, col]) < 1e-10:
            pivot_row_idx += 1
        
        if pivot_row_idx == m:
            col += 1
            continue 
        
        pivot_original_idx = row_indices_tracker[pivot_row_idx]

        if pivot_row_idx != rank:
            swap_desc = f'<b>Step {current_step_num}: Row Swap</b><br>Pivot found in row {pivot_row_idx + 1} (original row {pivot_original_idx + 1}). Swap R<sub>{rank + 1}</sub> ↔ R<sub>{pivot_row_idx + 1}</sub>.'
            A_copy[[rank, pivot_row_idx]] = A_copy[[pivot_row_idx, rank]]
            row_indices_tracker[rank], row_indices_tracker[pivot_row_idx] = row_indices_tracker[pivot_row_idx], row_indices_tracker[rank]
            current_pivots = [[p[0], p[1]] for p in pivot_indices]
            steps.append({
                'description': swap_desc,
                'matrix': format_matrix(A_copy),
                'all_pivot_coords': current_pivots 
            })
            current_step_num += 1
            pivot_original_idx = row_indices_tracker[rank]
            
        original_pivot_row_indices.append(pivot_original_idx)
        pivot_indices.append((rank, col))
        current_pivots = [[p[0], p[1]] for p in pivot_indices]

        pivot_val = A_copy[rank, col]
        if abs(pivot_val - 1.0) > 1e-10:
            norm_desc = f'<b>Step {current_step_num}: Normalize Pivot Row {rank + 1}</b><br>Divide R<sub>{rank + 1}</sub> by pivot {pivot_val:.4f}. <span class="formula">R<sub>{rank+1}</sub> → R<sub>{rank+1}</sub> / {pivot_val:.4f}</span>'
            A_copy[rank] = A_copy[rank] / pivot_val
            A_copy[rank, col] = 1.0
            steps.append({
                'description': norm_desc,
                'matrix': format_matrix(A_copy),
                'all_pivot_coords': current_pivots
            })
            current_step_num += 1
        
        elim_occurred = False
        elim_header = f'<b>Step {current_step_num}: Eliminate Below Pivot [{rank+1},{col+1}]</b>'
        elim_details = []
        for i in range(rank + 1, m):
            factor = A_copy[i, col]
            if abs(factor) > 1e-10:
                op_desc = f'&nbsp;&nbsp;<span class="formula">R<sub>{i+1}</sub> → R<sub>{i+1}</sub> - ({factor:.4f}) * R<sub>{rank+1}</sub></span>'
                A_copy[i] -= factor * A_copy[rank]
                A_copy[i, col] = 0.0
                elim_details.append(op_desc)
                elim_occurred = True
        
        if elim_occurred:
            steps.append({
                'description': elim_header + "<br>" + "<br>".join(elim_details),
                'matrix': format_matrix(A_copy),
                'all_pivot_coords': current_pivots
            })
            current_step_num += 1

        rank += 1
        col += 1
        
    steps.append({
        'description': f'<b>Step {current_step_num}: Row Echelon Form (REF) Reached</b><br>Rank = {rank}. Pivots are at positions: {[[p[0]+1, p[1]+1] for p in pivot_indices]}',
        'matrix': format_matrix(A_copy),
        'all_pivot_coords': [[p[0], p[1]] for p in pivot_indices] 
    })
    current_step_num += 1

    A_rref = A_copy
    for k in range(rank - 1, -1, -1):
        pivot_row, pivot_col = pivot_indices[k]
        elim_occurred_up = False
        elim_header_up = f'<b>Step {current_step_num}: Eliminate Above Pivot [{pivot_row+1},{pivot_col+1}]</b>'
        elim_details_up = []
        for i in range(pivot_row - 1, -1, -1):
            factor = A_rref[i, pivot_col]
            if abs(factor) > 1e-10:
                op_desc = f'&nbsp;&nbsp;<span class="formula">R<sub>{i+1}</sub> → R<sub>{i+1}</sub> - ({factor:.4f}) * R<sub>{pivot_row+1}</sub></span>'
                A_rref[i] -= factor * A_rref[pivot_row]
                A_rref[i, pivot_col] = 0.0
                elim_details_up.append(op_desc)
                elim_occurred_up = True
        
        if elim_occurred_up:
            current_pivots = [[p[0], p[1]] for p in pivot_indices]
            steps.append({
                'description': elim_header_up + "<br>" + "<br>".join(elim_details_up),
                'matrix': format_matrix(A_rref),
                'all_pivot_coords': current_pivots
            })
            current_step_num += 1

    steps.append({
        'description': f'<b>Step {current_step_num}: Reduced Row Echelon Form (RREF) Reached</b><br>Rank = {rank}.',
        'matrix': format_matrix(A_rref),
        'all_pivot_coords': [[p[0], p[1]] for p in pivot_indices] 
    })
    current_step_num += 1

    pivot_cols_list = [p[1] for p in pivot_indices]
    row_space_basis = A_orig[original_pivot_row_indices, :] if original_pivot_row_indices else []
    column_space_basis_cols = [A_orig[:, j] for j in pivot_cols_list] if pivot_cols_list else []
    null_space_basis = []
    free_cols = [j for j in range(n) if j not in pivot_cols_list]
    nullity = len(free_cols)
    for free_col in free_cols:
        x = np.zeros(n)
        x[free_col] = 1
        for i in range(rank):
            pivot_row, pivot_col = pivot_indices[i]
            x[pivot_col] = -A_rref[pivot_row, free_col]
        null_space_basis.append(x)
    rank_nullity_verified = (rank + nullity) == n

    formatted_row_basis = format_matrix(np.array(row_space_basis))
    formatted_col_basis = [format_matrix([col_vec])[0] for col_vec in column_space_basis_cols]
    formatted_null_basis = format_matrix(np.array(null_space_basis))

    desc_row = f"<b>Step {current_step_num}: Row Space Basis</b><br>"
    desc_row += f"(Using original matrix rows corresponding to pivot rows found: {', '.join(map(str, [idx+1 for idx in original_pivot_row_indices]))})<br>"
    desc_row += format_vectors_for_desc(formatted_row_basis, 'r')
    steps.append({'description': desc_row})
    current_step_num += 1
    
    desc_col = f"<b>Step {current_step_num}: Column Space Basis</b><br>"
    desc_col += f"(Using original matrix columns corresponding to pivot columns: {', '.join(map(str, [p+1 for p in pivot_cols_list]))})<br>"
    desc_col += format_vectors_for_desc(formatted_col_basis, 'c')
    steps.append({'description': desc_col})
    current_step_num += 1

    desc_null = f"<b>Step {current_step_num}: Null Space Basis (Ax=0)</b><br>"
    desc_null += f"Nullity = {nullity}. Basis vectors span the solution space of Ax=0.<br>"
    desc_null += format_vectors_for_desc(formatted_null_basis, 'n')
    steps.append({'description': desc_null})
    current_step_num += 1

    desc_summary = f"<b>Step {current_step_num}: Final Summary & Bases</b><br>"
    desc_summary += f"&nbsp;&nbsp;Rank(A) = {rank}<br>"
    desc_summary += f"&nbsp;&nbsp;Nullity(A) = {nullity}<br>"
    desc_summary += f"&nbsp;&nbsp;Rank + Nullity = {rank + nullity} (n={n}) {'(Verified)' if rank_nullity_verified else '(Verification Failed!)'}<hr>"
    desc_summary += "<b>Row Space Basis:</b><br>"
    desc_summary += format_vectors_for_desc(formatted_row_basis, 'r') + "<hr>"
    desc_summary += "<b>Column Space Basis:</b><br>"
    desc_summary += format_vectors_for_desc(formatted_col_basis, 'c') + "<hr>"
    desc_summary += "<b>Null Space Basis:</b><br>"
    desc_summary += format_vectors_for_desc(formatted_null_basis, 'n')
    steps.append({
        'description': desc_summary, 
        'is_final_summary': True 
    })

    return steps
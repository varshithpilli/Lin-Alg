document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded. Initializing script...'); // Check if this block runs

    // Initialize inverse matrix tab
    const inverseSize = document.getElementById('inverse-size');
    const createInverseBtn = document.getElementById('create-inverse-matrix');
    const calculateInverseBtn = document.getElementById('calculate-inverse');
    const inverseMatrixContainer = document.getElementById('inverse-matrix-container');
    const inverseResultContainer = document.getElementById('inverse-result');

    // Initialize Gram-Schmidt tab
    const orthoVectors = document.getElementById('ortho-vectors');
    const orthoDimension = document.getElementById('ortho-dimension');
    const createVectorsBtn = document.getElementById('create-vectors');
    const calculateOrthonormalBtn = document.getElementById('calculate-orthonormal');
    const vectorsContainer = document.getElementById('vectors-container');
    const orthoResultContainer = document.getElementById('ortho-result');

    // Initialize matrix spaces tab
    const basisRows = document.getElementById('basis-rows');
    const basisCols = document.getElementById('basis-cols');
    const createBasisBtn = document.getElementById('create-basis-matrix');
    const calculateBasesBtn = document.getElementById('calculate-bases');
    const basisMatrixContainer = document.getElementById('basis-matrix-container');
    const basisResultContainer = document.getElementById('basis-result');

    // Initialize REF tab
    const refRows = document.getElementById('ref-rows');
    const refCols = document.getElementById('ref-cols');
    const createRefMatrixBtn = document.getElementById('create-ref-matrix');
    const calculateRefBtn = document.getElementById('calculate-ref');
    const refMatrixContainer = document.getElementById('ref-matrix-container');
    const refResultContainer = document.getElementById('ref-result'); // Not used for steps, but maybe placeholder

    // Initialize RREF tab
    const rrefRows = document.getElementById('rref-rows');
    const rrefCols = document.getElementById('rref-cols');
    const createRrefMatrixBtn = document.getElementById('create-rref-matrix');
    const calculateRrefBtn = document.getElementById('calculate-rref');
    const rrefMatrixContainer = document.getElementById('rref-matrix-container');
    // const rrefResultContainer = document.getElementById('rref-result'); // Not used if using placeholder/steps

    // Initialize Equations tab
    const eqnsRows = document.getElementById('eqns-rows');
    const eqnsCols = document.getElementById('eqns-cols');
    const createEqnsMatrixBtn = document.getElementById('create-eqns-matrix');
    const calculateEqnsBtn = document.getElementById('calculate-eqns');
    const eqnsMatrixContainer = document.getElementById('eqns-matrix-container');
    // const eqnsResultContainer = document.getElementById('eqns-result'); // Not used

    // Shared state for inverse steps
    let inverseSteps = [];
    let currentInverseStepIndex = 0;

    // Get references to the new HTML elements
    const inverseResultStepsContainer = document.getElementById('inverse-result-steps');
    const inverseStepDescription = document.getElementById('inverse-step-description');
    const inverseStepMatrixContainer = document.getElementById('inverse-step-matrix');
    const inverseStepCounter = document.getElementById('inverse-step-counter');
    const inversePrevStepBtn = document.getElementById('inverse-prev-step');
    const inverseNextStepBtn = document.getElementById('inverse-next-step');
    const inverseShowSolutionBtn = document.getElementById('inverse-show-solution');
    const inverseResultPlaceholder = document.getElementById('inverse-result-placeholder');

    // --- Basis Section State and Elements ---
    let basisSteps = [];
    let currentBasisStepIndex = 0;

    // Get references to the new HTML elements
    const basisResultStepsContainer = document.getElementById('basis-result-steps');
    const basisStepDescription = document.getElementById('basis-step-description');
    const basisStepMatrixContainer = document.getElementById('basis-step-matrix');
    const basisStepCounter = document.getElementById('basis-step-counter');
    const basisPrevStepBtn = document.getElementById('basis-prev-step');
    const basisNextStepBtn = document.getElementById('basis-next-step');
    const basisShowSolutionBtn = document.getElementById('basis-show-solution');
    const basisResultPlaceholder = document.getElementById('basis-result-placeholder');

    // --- Gram-Schmidt Section State and Elements ---
    let orthoSteps = [];
    let currentOrthoStepIndex = 0;
    const orthoResultStepsContainer = document.getElementById('ortho-result-steps');
    const orthoStepDescription = document.getElementById('ortho-step-description');
    const orthoStepVectorsContainer = document.getElementById('ortho-step-vectors');
    const orthoStepCounter = document.getElementById('ortho-step-counter');
    const orthoPrevStepBtn = document.getElementById('ortho-prev-step');
    const orthoNextStepBtn = document.getElementById('ortho-next-step');
    const orthoShowSolutionBtn = document.getElementById('ortho-show-solution');
    const orthoResultPlaceholder = document.getElementById('ortho-result-placeholder');

    // --- REF Section State and Elements ---
    let refSteps = [];
    let currentRefStepIndex = 0;
    const refResultStepsContainer = document.getElementById('ref-result-steps');
    const refStepDescription = document.getElementById('ref-step-description');
    const refStepMatrixContainer = document.getElementById('ref-step-matrix');
    const refStepCounter = document.getElementById('ref-step-counter');
    const refPrevStepBtn = document.getElementById('ref-prev-step');
    const refNextStepBtn = document.getElementById('ref-next-step');
    const refShowSolutionBtn = document.getElementById('ref-show-solution');
    const refResultPlaceholder = document.getElementById('ref-result-placeholder');

    // --- RREF Section State and Elements ---
    let rrefSteps = [];
    let currentRrefStepIndex = 0;
    const rrefResultStepsContainer = document.getElementById('rref-result-steps');
    const rrefStepDescription = document.getElementById('rref-step-description');
    const rrefStepMatrixContainer = document.getElementById('rref-step-matrix');
    const rrefStepCounter = document.getElementById('rref-step-counter');
    const rrefPrevStepBtn = document.getElementById('rref-prev-step');
    const rrefNextStepBtn = document.getElementById('rref-next-step');
    const rrefShowSolutionBtn = document.getElementById('rref-show-solution');
    const rrefResultPlaceholder = document.getElementById('rref-result-placeholder');

    // --- Equations Section State and Elements ---
    let eqnsSteps = [];
    let currentEqnsStepIndex = 0;
    const eqnsResultStepsContainer = document.getElementById('eqns-result-steps');
    const eqnsStepDescription = document.getElementById('eqns-step-description');
    const eqnsStepMatrixContainer = document.getElementById('eqns-step-matrix');
    const eqnsStepCounter = document.getElementById('eqns-step-counter');
    const eqnsPrevStepBtn = document.getElementById('eqns-prev-step');
    const eqnsNextStepBtn = document.getElementById('eqns-next-step');
    const eqnsShowSolutionBtn = document.getElementById('eqns-show-solution');
    const eqnsResultPlaceholder = document.getElementById('eqns-result-placeholder');

    // --- Add Checks: Log if elements are found --- 
    console.log('Checking Gram-Schmidt button elements:');
    console.log('orthoPrevStepBtn found:', !!orthoPrevStepBtn); // Log true if found, false if null
    console.log('orthoNextStepBtn found:', !!orthoNextStepBtn);
    console.log('orthoShowSolutionBtn found:', !!orthoShowSolutionBtn);
    // --- End Checks ---

    // Event listeners for creating matrices
    createInverseBtn.addEventListener('click', () => {
        const size = parseInt(inverseSize.value);
        if (size > 0) { // Basic validation
             createMatrix(inverseMatrixContainer, size, size, 'inverse');
        } else {
             alert("Please enter a valid matrix size.");
        }
    });
    createBasisBtn.addEventListener('click', () => createMatrix(basisMatrixContainer, basisRows.value, basisCols.value, 'basis'));
    createVectorsBtn.addEventListener('click', () => createVectors(vectorsContainer, orthoVectors.value, orthoDimension.value));
    createRefMatrixBtn.addEventListener('click', () => createMatrix(refMatrixContainer, refRows.value, refCols.value, 'ref'));
    createRrefMatrixBtn.addEventListener('click', () => createMatrix(rrefMatrixContainer, rrefRows.value, rrefCols.value, 'rref'));
    createEqnsMatrixBtn.addEventListener('click', () => createMatrix(eqnsMatrixContainer, eqnsRows.value, eqnsCols.value, 'eqns'));

    // Event listeners for calculations
    calculateInverseBtn.addEventListener('click', calculateInverse);
    calculateOrthonormalBtn.addEventListener('click', calculateOrthonormal);
    calculateBasesBtn.addEventListener('click', calculateBases);
    calculateRefBtn.addEventListener('click', calculateRef);
    calculateRrefBtn.addEventListener('click', calculateRref);
    calculateEqnsBtn.addEventListener('click', calculateEqns);

    // Create default matrices on page load
    const defaultSize = parseInt(inverseSize.value) || 3; // Use value from input or default to 3
    createMatrix(inverseMatrixContainer, defaultSize, defaultSize, 'inverse');
    createMatrix(basisMatrixContainer, 3, 3, 'basis');
    createVectors(vectorsContainer, 3, 3);
    createMatrix(refMatrixContainer, 3, 3, 'ref'); // Default REF matrix
    createMatrix(rrefMatrixContainer, 3, 3, 'rref'); // Default RREF matrix
    createMatrix(eqnsMatrixContainer, 3, 4, 'eqns'); // Default Equations matrix (e.g., 3x4 augmented)

    // Matrix creation function
    function createMatrix(container, rows, cols, prefix) {
        container.innerHTML = '';
        let table = document.createElement('table');
        table.className = 'matrix-table';
        table.id = `${prefix}-table`;
        
        rows = parseInt(rows) || 1; 
        cols = parseInt(cols) || 1;
        
        for (let i = 0; i < rows; i++) {
            let row = document.createElement('tr');
            for (let j = 0; j < cols; j++) { 
                let cell = document.createElement('td');
                // Clear previous separator classes potentially added
                cell.classList.remove('rhs-separator'); 
                cell.classList.remove('augmented-input-separator'); 

                let input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.id = `${prefix}-${i}-${j}`;

                // Set default values
                if (prefix === 'eqns') {
                    if (j === cols - 1) { // Last column (constants)
                        input.value = 0; 
                    } else { // Coefficient part
                        input.value = (i === j) ? 1 : 0; 
                    }
                } else { 
                     input.value = (i === j) ? 1 : 0; 
                }
                                
                cell.appendChild(input);
                row.appendChild(cell);

                // Add separator CLASS to the cell BEFORE the last column for equations
                if (prefix === 'eqns' && j === cols - 2) { 
                    cell.classList.add('rhs-separator');
                }
            }
            table.appendChild(row);
        }
        container.appendChild(table);
    }

    // Vectors creation function
    function createVectors(container, numVectors, dimension) {
        container.innerHTML = '';
        
        for (let v = 0; v < numVectors; v++) {
            let vectorDiv = document.createElement('div');
            vectorDiv.className = 'vector-container mb-3';
            
            let vectorLabel = document.createElement('div');
            vectorLabel.className = 'vector-label mb-2';
            vectorLabel.textContent = `Vector ${v + 1}:`;
            vectorDiv.appendChild(vectorLabel);
            
            let table = document.createElement('table');
            table.className = 'matrix-table';
            let row = document.createElement('tr');
            
            for (let i = 0; i < dimension; i++) {
                let cell = document.createElement('td');
                let input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.id = `vector-${v}-${i}`;
                input.value = v === i ? 1 : 0; // Default values
                cell.appendChild(input);
                row.appendChild(cell);
            }
            
            table.appendChild(row);
            vectorDiv.appendChild(table);
            container.appendChild(vectorDiv);
        }
    }

    // Get matrix values from input fields
    function getMatrixFromTable(prefix, rows, cols) {
        let matrix = [];
        for (let i = 0; i < rows; i++) {
            let row = [];
            for (let j = 0; j < cols; j++) {
                const input = document.getElementById(`${prefix}-${i}-${j}`);
                row.push(parseFloat(input.value) || 0);
            }
            matrix.push(row);
        }
        return matrix;
    }

    // Get vectors from input fields
    function getVectorsFromInputs(numVectors, dimension) {
        let vectors = [];
        for (let v = 0; v < numVectors; v++) {
            let vector = [];
            for (let i = 0; i < dimension; i++) {
                const input = document.getElementById(`vector-${v}-${i}`);
                vector.push(parseFloat(input.value) || 0);
            }
            vectors.push(vector);
        }
        return vectors;
    }

    // Display matrix result (Modified for cumulative pivot highlighting)
    function displayMatrixResult(container, matrix, highlight_coords = [], split_col_index = null) {
        console.log('[displayMatrixResult] Called. Highlight Coords:', highlight_coords, 'Split Index:', split_col_index); // Log received parameters
        container.innerHTML = '';
        
        if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])) {
            return; 
        }

        const resultMatrix = document.createElement('div');
        resultMatrix.className = 'result-matrix';
        
        // Create a Set of highlight coordinates strings for efficient lookup
        const highlightSet = new Set((highlight_coords || []).map(coord => {
            if (Array.isArray(coord) && coord.length === 2) { // Ensure coord is valid [row, col]
                return `${coord[0]},${coord[1]}`;
            } 
            console.warn('[displayMatrixResult] Invalid coordinate format in highlight_coords:', coord); // Warn about invalid coords
            return null; // Return null for invalid items
        }).filter(item => item !== null)); // Filter out nulls
        console.log('[displayMatrixResult] Highlight Set created:', highlightSet);

        for (let i = 0; i < matrix.length; i++) {
            const row = document.createElement('div');
            row.className = 'result-matrix-row';
            
            for (let j = 0; j < matrix[i].length; j++) {
                const cell = document.createElement('div');
                cell.className = 'result-matrix-cell';
                const currentCoordStr = `${i},${j}`;
                
                // Add highlighting class if current cell coordinates are in the set
                if (highlightSet.has(currentCoordStr)) {
                    console.log(`[displayMatrixResult] Applying highlight to [${i}, ${j}]`); // Log highlight application
                    cell.classList.add('pivot-highlight');
                } else {
                    cell.classList.remove('pivot-highlight'); // Explicitly remove if not in set
                }
                
                // Add separator class if this is the cell before the split 
                if (split_col_index !== null && j === split_col_index - 1) {
                    cell.classList.add('augmented-separator');
                } else {
                    cell.classList.remove('augmented-separator'); // Explicitly remove if not needed
                }

                // Handle potential complex numbers formatted as dicts
                let valueText = '';
                const cellValue = matrix[i][j];
                if (typeof cellValue === 'object' && cellValue !== null && 'real' in cellValue) {
                     if (Math.abs(cellValue.imag) < 1e-9) { // Treat near-zero imaginary as real
                        valueText = `${cellValue.real}`;
                    } else {
                        valueText = `${cellValue.real}${cellValue.imag >= 0 ? '+' : ''}${cellValue.imag}i`;
                    }
                } else {
                    // Assume real number
                    valueText = `${cellValue}`;
                }
                cell.textContent = valueText;
                row.appendChild(cell);
            }
            resultMatrix.appendChild(row);
        }
        container.appendChild(resultMatrix);
    }

    // Display vectors result (Modified for specific indices)
    function displayVectorsResult(container, vectors, message, titlePrefix = 'Vector', indices = []) {
        container.innerHTML = ''; // Clear previous content
        
        // Basic check for valid vectors array
        if (!Array.isArray(vectors)) {
            container.innerHTML = '<div class="alert alert-info">Invalid vector data for this step.</div>';
            return;
        }
        // Handle case where vectors might be empty (e.g., for info steps or empty basis)
        if (vectors.length === 0) {
            // Display the message if provided, otherwise indicate no vectors
            if(message) {
                container.innerHTML = `<div class="alert alert-info">${message}</div>`;
            } else {
                 container.innerHTML = '<div class="alert alert-info">No vectors to display for this step.</div>';
            }
            return;
        }
        // Check if the first element is an array (basic check for list of lists)
        if (!Array.isArray(vectors[0])) {
             container.innerHTML = '<div class="alert alert-warning">Vector data format seems incorrect for this step.</div>';
            return;
        }

        vectors.forEach((vector, v) => {
            if (!Array.isArray(vector)) return; // Skip if not a valid vector

            const vectorDiv = document.createElement('div');
            vectorDiv.className = 'result-vector mb-3';
            
            const vectorLabel = document.createElement('div');
            vectorLabel.className = 'mb-1 fw-bold';
            
            // *** Use the provided index if available, otherwise use the loop index v ***
            const displayIndex = (indices && indices.length > v) ? indices[v] : (v + 1);
            vectorLabel.innerHTML = `${titlePrefix}<sub>${displayIndex}</sub>:`; // Use subscript for index
            vectorDiv.appendChild(vectorLabel);
            
            const row = document.createElement('div');
            row.className = 'result-matrix-row d-flex flex-wrap'; 
            
            vector.forEach(val => {
                const cell = document.createElement('div');
                cell.className = 'result-matrix-cell me-2 mb-1'; 
                let value = Number(val); // Already formatted by backend
                // If backend formatting is removed, add rounding back here:
                // let value = Number(val.toFixed(5)); 
                // if (Math.abs(value) < 1e-10) value = 0;
                cell.textContent = value;
                row.appendChild(cell);
            });
            
            vectorDiv.appendChild(row);
            container.appendChild(vectorDiv);
        });
        
        // Append message if provided (even if vectors were displayed)
        if (message) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'result-message mt-2 alert alert-info';
            messageDiv.textContent = message;
            container.appendChild(messageDiv);
        }
    }

    // Calculate matrix inverse (Modified for Steps)
    function calculateInverse() {
        // Get size directly from the single input
        const size = parseInt(inverseSize.value);

        // Basic validation for size
        if (isNaN(size) || size <= 0) {
             inverseResultPlaceholder.innerHTML = '<div class="alert alert-warning">Please enter a valid positive matrix size.</div>';
             inverseResultStepsContainer.style.display = 'none'; 
             inverseResultPlaceholder.style.display = 'block'; 
            return;
        }
        
        // Use size for both rows and columns when getting matrix data
        const matrix = getMatrixFromTable('inverse', size, size); 

        // Show loading indicator (optional)
        inverseResultPlaceholder.innerHTML = '<div class="placeholder-text">Calculating...</div>';
        inverseResultStepsContainer.style.display = 'none';
        inverseResultPlaceholder.style.display = 'block';

        fetch('https://lin-alg.onrender.com/calculate/inverse', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ matrix })
        })
        .then(response => {
             if (!response.ok) {
                 // Try to parse error from response body
                 return response.json().then(errData => {
                     throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                 }).catch(() => {
                     // Fallback if error parsing fails
                     throw new Error(`HTTP error! status: ${response.status}`);
                 });
             }
             return response.json();
         })
        .then(data => {
            if (data.success && data.steps && data.steps.length > 0) {
                inverseSteps = data.steps;
                currentInverseStepIndex = 0;
                displayInverseStep(currentInverseStepIndex);
                inverseResultStepsContainer.style.display = 'block'; // Show step container
                inverseResultPlaceholder.style.display = 'none'; // Hide placeholder
            } else {
                 const errorMsg = data.error || 'Calculation failed or returned no steps.';
                 inverseResultPlaceholder.innerHTML = `<div class="alert alert-danger">${errorMsg}</div>`;
                 inverseResultStepsContainer.style.display = 'none';
                 inverseResultPlaceholder.style.display = 'block';
            }
        })
        .catch(error => {
             console.error("Fetch Error:", error);
             inverseResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
             inverseResultStepsContainer.style.display = 'none';
             inverseResultPlaceholder.style.display = 'block';
        });
    }

    // Function to display a specific INVERSE step
    function displayInverseStep(index) {
        if (index < 0 || index >= inverseSteps.length) return;

        const step = inverseSteps[index];
        inverseStepDescription.innerHTML = step.description || `Step ${index + 1}`;
        
        // *** Pass the split column index and pivot coord to displayMatrixResult ***
        displayMatrixResult(
            inverseStepMatrixContainer, 
            step.matrix, 
            step.pivot_coord || null, // Pass pivot coord if exists
            step.split_col_index || null // Pass split index if exists
        );

        // Update counter
        inverseStepCounter.textContent = `Step ${index + 1} / ${inverseSteps.length}`;

        // Update button states
        inversePrevStepBtn.disabled = index === 0;
        inverseNextStepBtn.disabled = index === inverseSteps.length - 1;
        inverseShowSolutionBtn.disabled = index === inverseSteps.length - 1;
    }

    // Event listeners for step navigation
    inversePrevStepBtn.addEventListener('click', () => {
        if (currentInverseStepIndex > 0) {
            currentInverseStepIndex--;
            displayInverseStep(currentInverseStepIndex);
        }
    });

    inverseNextStepBtn.addEventListener('click', () => {
        if (currentInverseStepIndex < inverseSteps.length - 1) {
            currentInverseStepIndex++;
            displayInverseStep(currentInverseStepIndex);
        }
    });

    inverseShowSolutionBtn.addEventListener('click', () => {
        currentInverseStepIndex = inverseSteps.length - 1;
        displayInverseStep(currentInverseStepIndex);
        // Optionally, highlight the final inverse if stored separately
        const finalStep = inverseSteps[currentInverseStepIndex];
        if(finalStep.final_inverse) {
            // Clear current step matrix and display only the final one
            inverseStepMatrixContainer.innerHTML = '<h5>Final Inverse Matrix (A⁻¹):</h5>';
            displayMatrixResult(inverseStepMatrixContainer, finalStep.final_inverse);
        }
    });

    // Calculate orthonormal basis using Gram-Schmidt
    function calculateOrthonormal() {
        const numVectors = parseInt(orthoVectors.value);
        const dimension = parseInt(orthoDimension.value);
        const vectors = getVectorsFromInputs(numVectors, dimension);
        
        // Basic validation
        if (isNaN(numVectors) || numVectors <= 0 || isNaN(dimension) || dimension <= 0) {
            orthoResultPlaceholder.innerHTML = '<div class="alert alert-warning">Invalid vector dimensions or count.</div>';
            orthoResultStepsContainer.style.display = 'none';
            orthoResultPlaceholder.style.display = 'block';
            return;
        }
        // Check if vectors are linearly independent (optional, backend should handle this)
        // Placeholder for calculation
        orthoResultPlaceholder.innerHTML = '<div class="placeholder-text">Calculating...</div>';
        orthoResultStepsContainer.style.display = 'none';
        orthoResultPlaceholder.style.display = 'block';

        fetch('https://lin-alg.onrender.com/calculate/orthonormal', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ vectors })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                orthoResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                orthoResultStepsContainer.style.display = 'none';
                orthoResultPlaceholder.style.display = 'block';
            } else if (data.steps && data.steps.length > 0) {
                orthoSteps = data.steps; // Assuming backend returns { steps: [...] }
                currentOrthoStepIndex = 0;
                displayOrthoStep(currentOrthoStepIndex);
                orthoResultStepsContainer.style.display = 'block'; // Show steps container
                orthoResultPlaceholder.style.display = 'none'; // Hide placeholder
            } else {
                orthoResultPlaceholder.innerHTML = '<div class="alert alert-warning">Calculation finished, but no steps were returned.</div>';
                orthoResultStepsContainer.style.display = 'none';
                orthoResultPlaceholder.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error calculating orthonormal basis:', error);
            orthoResultPlaceholder.innerHTML = `<div class="alert alert-danger">An error occurred: ${error}. Check the console for details.</div>`;
            orthoResultStepsContainer.style.display = 'none';
            orthoResultPlaceholder.style.display = 'block';
        });
    }

    // Display Gram-Schmidt Step (New Function)
    function displayOrthoStep(index) {
        console.log(`displayOrthoStep called with index: ${index}`);
        if (index < 0 || index >= orthoSteps.length) {
            console.error("Invalid step index for Gram-Schmidt:", index, "Total steps:", orthoSteps.length);
            return;
        }

        const step = orthoSteps[index];
        console.log('Step data:', step);

        // Assuming each step has format: { description: "...", vectors: [[...], [...]], ?vector_type: "u" or "e" }
        const description = step.description || `Step ${index + 1}`; 
        console.log('Setting description:', description);
        orthoStepDescription.innerHTML = description;
        
        // Use the displayVectorsResult function to render the vectors for the current step
        let titlePrefix = 'Vector';
        if (step.vector_type === 'u') {
            titlePrefix = 'u';
        } else if (step.vector_type === 'e') {
            titlePrefix = 'e';
        } else if (step.vector_type === 'v') {
            titlePrefix = 'v'; 
        }
        // Pass the specific indices from the step data
        const vectorIndices = step.vector_indices || []; 
        console.log('Calling displayVectorsResult with prefix:', titlePrefix, 'indices:', vectorIndices, 'and vectors:', step.vectors);
        displayVectorsResult(orthoStepVectorsContainer, step.vectors, null, titlePrefix, vectorIndices);

        // Update counter
        const counterText = `Step ${index + 1} / ${orthoSteps.length}`;
        console.log('Setting counter:', counterText);
        orthoStepCounter.textContent = counterText;

        // Update button states
        console.log('Updating button states. Prev disabled:', index === 0, 'Next disabled:', index === orthoSteps.length - 1);
        orthoPrevStepBtn.disabled = index === 0;
        orthoNextStepBtn.disabled = index === orthoSteps.length - 1;
        orthoShowSolutionBtn.disabled = index === orthoSteps.length - 1; // Disable if already on last step
        console.log('displayOrthoStep finished for index:', index);
    }

    // Gram-Schmidt (New Listeners)
    if (orthoPrevStepBtn) { // Only attach if element exists
        orthoPrevStepBtn.addEventListener('click', () => {
            console.log('Ortho Prev button clicked. Current index:', currentOrthoStepIndex);
            if (currentOrthoStepIndex > 0) {
                currentOrthoStepIndex--;
                console.log('Ortho Prev: New index:', currentOrthoStepIndex);
                displayOrthoStep(currentOrthoStepIndex);
            }
        });
    } else { 
        console.error('Could not attach listener: orthoPrevStepBtn not found'); 
    }

    if (orthoNextStepBtn) { // Only attach if element exists
        orthoNextStepBtn.addEventListener('click', () => {
            console.log('Ortho Next button clicked. Current index:', currentOrthoStepIndex, 'Total steps:', orthoSteps.length);
            if (currentOrthoStepIndex < orthoSteps.length - 1) {
                currentOrthoStepIndex++;
                console.log('Ortho Next: New index:', currentOrthoStepIndex);
                displayOrthoStep(currentOrthoStepIndex);
            }
        });
    } else { 
        console.error('Could not attach listener: orthoNextStepBtn not found'); 
    }

    if (orthoShowSolutionBtn) { // Only attach if element exists
        orthoShowSolutionBtn.addEventListener('click', () => {
            console.log('Ortho Show Solution button clicked. Total steps:', orthoSteps.length);
            currentOrthoStepIndex = orthoSteps.length - 1;
            console.log('Ortho Show Solution: Setting index to:', currentOrthoStepIndex);
            displayOrthoStep(currentOrthoStepIndex);
        });
    } else { 
        console.error('Could not attach listener: orthoShowSolutionBtn not found'); 
    }

    // Calculate matrix spaces (Modified for Steps)
    function calculateBases() {
        const rows = parseInt(basisRows.value);
        const cols = parseInt(basisCols.value);
        const matrix = getMatrixFromTable('basis', rows, cols);

        // Show loading indicator
        basisResultPlaceholder.innerHTML = '<div class="placeholder-text">Calculating...</div>';
        basisResultStepsContainer.style.display = 'none';
        basisResultPlaceholder.style.display = 'block';

        fetch('https://lin-alg.onrender.com/calculate/bases', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ matrix })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                }).catch(() => {
                    throw new Error(`HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                basisResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                basisResultStepsContainer.style.display = 'none';
                basisResultPlaceholder.style.display = 'block';
            } else if (data.steps && data.steps.length > 0) {
                basisSteps = data.steps;
                currentBasisStepIndex = 0;
                displayBasisStep(currentBasisStepIndex);
                basisResultStepsContainer.style.display = 'block'; // Show step container
                basisResultPlaceholder.style.display = 'none'; // Hide placeholder
            } else {
                basisResultPlaceholder.innerHTML = '<div class="alert alert-warning">Calculation finished, but no steps were returned.</div>';
                basisResultStepsContainer.style.display = 'none';
                basisResultPlaceholder.style.display = 'block';
            }
        })
        .catch(error => {
            console.error("Fetch Error for Bases:", error);
            basisResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            basisResultStepsContainer.style.display = 'none';
            basisResultPlaceholder.style.display = 'block';
        });
    }

    // Function to display a specific basis step
    function displayBasisStep(index) {
        console.log(`displayBasisStep called with index: ${index}`); // Debug log
        if (index < 0 || index >= basisSteps.length) {
             console.error("Invalid step index for Basis:", index, "Total steps:", basisSteps.length);
             return;
        }

        const step = basisSteps[index];
        console.log('Basis Step data:', step); // Debug log
        
        // Set description
        const description = step.description || `Step ${index + 1}`; 
        console.log('Setting basis description:', description); // Debug log
        basisStepDescription.innerHTML = description;

        // Check if this step includes a matrix to display
        if (step.matrix && Array.isArray(step.matrix)) {
            console.log(`[displayBasisStep] Displaying matrix for step ${index}. Passing all_pivot_coords:`, step.all_pivot_coords);
            basisStepMatrixContainer.style.display = 'block'; 
            // Pass the list of all pivot coordinates, ensure it's an array
            displayMatrixResult(basisStepMatrixContainer, step.matrix, step.all_pivot_coords || []); 
        } else {
            console.log(`[displayBasisStep] Hiding matrix container for step ${index}.`);
            basisStepMatrixContainer.innerHTML = '';
            basisStepMatrixContainer.style.display = 'none'; 
        }

        // Update counter
        const counterText = `Step ${index + 1} / ${basisSteps.length}`;
        console.log('Setting basis counter:', counterText); // Debug log
        basisStepCounter.textContent = counterText;

        // Update button states
        console.log('Updating basis button states. Prev disabled:', index === 0, 'Next disabled:', index === basisSteps.length - 1); // Debug log
        basisPrevStepBtn.disabled = index === 0;
        basisNextStepBtn.disabled = index === basisSteps.length - 1;
        
        // *** Re-enable the Show Solution button and manage its state ***
        basisShowSolutionBtn.disabled = index === basisSteps.length - 1; // Disable if on last step
        basisShowSolutionBtn.style.display = 'inline-block'; // Make sure it's visible
        
        console.log('displayBasisStep finished for index:', index); // Debug log
    }

    // Event listeners for basis step navigation
    basisPrevStepBtn.addEventListener('click', () => {
        if (currentBasisStepIndex > 0) {
            currentBasisStepIndex--;
            displayBasisStep(currentBasisStepIndex);
        }
    });

    basisNextStepBtn.addEventListener('click', () => {
        if (currentBasisStepIndex < basisSteps.length - 1) {
            currentBasisStepIndex++;
            displayBasisStep(currentBasisStepIndex);
        }
    });

    // *** Re-add the listener for the Show Solution button ***
    if (basisShowSolutionBtn) {
         basisShowSolutionBtn.addEventListener('click', () => {
             console.log('Basis Show Solution button clicked. Total steps:', basisSteps.length);
             currentBasisStepIndex = basisSteps.length - 1;
             console.log('Basis Show Solution: Setting index to:', currentBasisStepIndex);
             displayBasisStep(currentBasisStepIndex);
         });
     } else {
         console.error('Could not attach listener: basisShowSolutionBtn not found'); 
     }

    // Calculate REF (New Function)
    function calculateRef() {
        const rows = parseInt(refRows.value);
        const cols = parseInt(refCols.value);
        
        if (isNaN(rows) || rows <= 0 || isNaN(cols) || cols <= 0) {
            refResultPlaceholder.innerHTML = '<div class="alert alert-warning">Please enter valid positive matrix dimensions.</div>';
            refResultStepsContainer.style.display = 'none';
            refResultPlaceholder.style.display = 'block';
            return;
        }

        const matrix = getMatrixFromTable('ref', rows, cols);

        // Show loading indicator
        refResultPlaceholder.innerHTML = '<div class="placeholder-text">Calculating...</div>';
        refResultStepsContainer.style.display = 'none';
        refResultPlaceholder.style.display = 'block';

        fetch('https://lin-alg.onrender.com/calculate/ref', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ matrix })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                }).catch(() => {
                    throw new Error(`HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                refResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                refResultStepsContainer.style.display = 'none';
                refResultPlaceholder.style.display = 'block';
            } else if (data.steps && data.steps.length > 0) {
                refSteps = data.steps;
                currentRefStepIndex = 0;
                displayRefStep(currentRefStepIndex);
                refResultStepsContainer.style.display = 'block'; // Show step container
                refResultPlaceholder.style.display = 'none'; // Hide placeholder
            } else {
                refResultPlaceholder.innerHTML = '<div class="alert alert-warning">Calculation finished, but no steps were returned.</div>';
                refResultStepsContainer.style.display = 'none';
                refResultPlaceholder.style.display = 'block';
            }
        })
        .catch(error => {
            console.error("Fetch Error for REF:", error);
            refResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            refResultStepsContainer.style.display = 'none';
            refResultPlaceholder.style.display = 'block';
        });
    }

    // Function to display a specific REF step (New Function)
    function displayRefStep(index) {
        if (index < 0 || index >= refSteps.length) return;

        const step = refSteps[index];
        refStepDescription.innerHTML = step.description || `Step ${index + 1}`;
        
        // Display matrix, passing the single pivot coord for highlighting during that step,
        // or all pivots for the final step description.
        const pivotCoordsToHighlight = step.pivot_coord ? [step.pivot_coord] : (index === refSteps.length - 1 ? step.all_pivot_coords : []);
        
        displayMatrixResult(
            refStepMatrixContainer, 
            step.matrix, 
            pivotCoordsToHighlight || [] // Pass pivot coord(s) if they exist
        );

        // Update counter
        refStepCounter.textContent = `Step ${index + 1} / ${refSteps.length}`;

        // Update button states
        refPrevStepBtn.disabled = index === 0;
        refNextStepBtn.disabled = index === refSteps.length - 1;
        refShowSolutionBtn.disabled = index === refSteps.length - 1;
    }

    // Event listeners for REF step navigation (New Listeners)
    if (refPrevStepBtn) {
        refPrevStepBtn.addEventListener('click', () => {
            if (currentRefStepIndex > 0) {
                currentRefStepIndex--;
                displayRefStep(currentRefStepIndex);
            }
        });
    }
    if (refNextStepBtn) {
        refNextStepBtn.addEventListener('click', () => {
            if (currentRefStepIndex < refSteps.length - 1) {
                currentRefStepIndex++;
                displayRefStep(currentRefStepIndex);
            }
        });
    }
    if (refShowSolutionBtn) {
        refShowSolutionBtn.addEventListener('click', () => {
            currentRefStepIndex = refSteps.length - 1;
            displayRefStep(currentRefStepIndex);
        });
    }

    // Calculate RREF (New Function)
    function calculateRref() {
        const rows = parseInt(rrefRows.value);
        const cols = parseInt(rrefCols.value);
        
        if (isNaN(rows) || rows <= 0 || isNaN(cols) || cols <= 0) {
            rrefResultPlaceholder.innerHTML = '<div class="alert alert-warning">Please enter valid positive matrix dimensions.</div>';
            rrefResultStepsContainer.style.display = 'none';
            rrefResultPlaceholder.style.display = 'block';
            return;
        }

        const matrix = getMatrixFromTable('rref', rows, cols);

        // Show loading indicator
        rrefResultPlaceholder.innerHTML = '<div class="placeholder-text">Calculating...</div>';
        rrefResultStepsContainer.style.display = 'none';
        rrefResultPlaceholder.style.display = 'block';

        fetch('https://lin-alg.onrender.com/calculate/rref', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ matrix })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                }).catch(() => {
                    throw new Error(`HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                rrefResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                rrefResultStepsContainer.style.display = 'none';
                rrefResultPlaceholder.style.display = 'block';
            } else if (data.steps && data.steps.length > 0) {
                rrefSteps = data.steps;
                currentRrefStepIndex = 0;
                displayRrefStep(currentRrefStepIndex);
                rrefResultStepsContainer.style.display = 'block'; // Show step container
                rrefResultPlaceholder.style.display = 'none'; // Hide placeholder
            } else {
                rrefResultPlaceholder.innerHTML = '<div class="alert alert-warning">Calculation finished, but no steps were returned.</div>';
                rrefResultStepsContainer.style.display = 'none';
                rrefResultPlaceholder.style.display = 'block';
            }
        })
        .catch(error => {
            console.error("Fetch Error for RREF:", error);
            rrefResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            rrefResultStepsContainer.style.display = 'none';
            rrefResultPlaceholder.style.display = 'block';
        });
    }

    // Function to display a specific RREF step (New Function)
    function displayRrefStep(index) {
        if (index < 0 || index >= rrefSteps.length) return;

        const step = rrefSteps[index];
        rrefStepDescription.innerHTML = step.description || `Step ${index + 1}`;
        
        // Highlight the active pivot for the step, or all pivots on the final step
        const pivotCoordsToHighlight = step.pivot_coord ? [step.pivot_coord] : (index === rrefSteps.length - 1 ? step.all_pivot_coords : []);
        
        displayMatrixResult(
            rrefStepMatrixContainer, 
            step.matrix, 
            pivotCoordsToHighlight || [] // Pass pivot coord(s)
        );

        // Update counter
        rrefStepCounter.textContent = `Step ${index + 1} / ${rrefSteps.length}`;

        // Update button states
        rrefPrevStepBtn.disabled = index === 0;
        rrefNextStepBtn.disabled = index === rrefSteps.length - 1;
        rrefShowSolutionBtn.disabled = index === rrefSteps.length - 1;
    }

    // Event listeners for RREF step navigation (New Listeners)
    if (rrefPrevStepBtn) {
        rrefPrevStepBtn.addEventListener('click', () => {
            if (currentRrefStepIndex > 0) {
                currentRrefStepIndex--;
                displayRrefStep(currentRrefStepIndex);
            }
        });
    }
    if (rrefNextStepBtn) {
        rrefNextStepBtn.addEventListener('click', () => {
            if (currentRrefStepIndex < rrefSteps.length - 1) {
                currentRrefStepIndex++;
                displayRrefStep(currentRrefStepIndex);
            }
        });
    }
    if (rrefShowSolutionBtn) {
        rrefShowSolutionBtn.addEventListener('click', () => {
            currentRrefStepIndex = rrefSteps.length - 1;
            displayRrefStep(currentRrefStepIndex);
        });
    }

    // Calculate Equation Solutions (New Function)
    function calculateEqns() {
        const rows = parseInt(eqnsRows.value);
        const cols = parseInt(eqnsCols.value);
        
        if (isNaN(rows) || rows <= 0 || isNaN(cols) || cols < 2) { // Need at least 2 columns for augmented matrix
            eqnsResultPlaceholder.innerHTML = '<div class="alert alert-warning">Please enter valid dimensions (at least 2 columns).</div>';
            eqnsResultStepsContainer.style.display = 'none';
            eqnsResultPlaceholder.style.display = 'block';
            return;
        }

        const matrix = getMatrixFromTable('eqns', rows, cols);

        // Show loading indicator
        eqnsResultPlaceholder.innerHTML = '<div class="placeholder-text">Calculating RREF and solving...</div>';
        eqnsResultStepsContainer.style.display = 'none';
        eqnsResultPlaceholder.style.display = 'block';

        fetch('https://lin-alg.onrender.com/calculate/equations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ matrix })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.error || `HTTP error! status: ${response.status}`);
                }).catch(() => {
                    throw new Error(`HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                eqnsResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                eqnsResultStepsContainer.style.display = 'none';
                eqnsResultPlaceholder.style.display = 'block';
            } else if (data.steps && data.steps.length > 0) {
                eqnsSteps = data.steps; // Combined RREF + solution steps
                currentEqnsStepIndex = 0;
                displayEqnsStep(currentEqnsStepIndex);
                eqnsResultStepsContainer.style.display = 'block'; // Show step container
                eqnsResultPlaceholder.style.display = 'none'; // Hide placeholder
            } else {
                eqnsResultPlaceholder.innerHTML = '<div class="alert alert-warning">Calculation finished, but no steps were returned.</div>';
                eqnsResultStepsContainer.style.display = 'none';
                eqnsResultPlaceholder.style.display = 'block';
            }
        })
        .catch(error => {
            console.error("Fetch Error for Equations:", error);
            eqnsResultPlaceholder.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            eqnsResultStepsContainer.style.display = 'none';
            eqnsResultPlaceholder.style.display = 'block';
        });
    }

    // Function to display a specific Equation Solving step (New Function)
    function displayEqnsStep(index) {
        if (index < 0 || index >= eqnsSteps.length) return;

        const step = eqnsSteps[index];
        eqnsStepDescription.innerHTML = step.description || `Step ${index + 1}`;
        
        // Highlight the active pivot for the RREF step, or all pivots on later steps
        const pivotCoordsToHighlight = step.pivot_coord ? [step.pivot_coord] : step.all_pivot_coords;
        
        // Always display the matrix for context, indicate augmentation line
        const numCols = step.matrix[0] ? step.matrix[0].length : 0;
        const splitColIndex = numCols > 0 ? numCols - 1 : null; // Index BEFORE the last column

        displayMatrixResult(
            eqnsStepMatrixContainer, 
            step.matrix, 
            pivotCoordsToHighlight || [],
            splitColIndex // Pass index to show augmentation line
        );

        // Update counter
        eqnsStepCounter.textContent = `Step ${index + 1} / ${eqnsSteps.length}`;

        // Update button states
        eqnsPrevStepBtn.disabled = index === 0;
        eqnsNextStepBtn.disabled = index === eqnsSteps.length - 1;
        eqnsShowSolutionBtn.disabled = index === eqnsSteps.length - 1;
    }

    // Event listeners for Equation Solving step navigation (New Listeners)
    if (eqnsPrevStepBtn) {
        eqnsPrevStepBtn.addEventListener('click', () => {
            if (currentEqnsStepIndex > 0) {
                currentEqnsStepIndex--;
                displayEqnsStep(currentEqnsStepIndex);
            }
        });
    }
    if (eqnsNextStepBtn) {
        eqnsNextStepBtn.addEventListener('click', () => {
            if (currentEqnsStepIndex < eqnsSteps.length - 1) {
                currentEqnsStepIndex++;
                displayEqnsStep(currentEqnsStepIndex);
            }
        });
    }
    if (eqnsShowSolutionBtn) {
        eqnsShowSolutionBtn.addEventListener('click', () => {
            currentEqnsStepIndex = eqnsSteps.length - 1;
            displayEqnsStep(currentEqnsStepIndex);
        });
    }

    // Display basis results (Original - Keep for reference or remove)
    /* function displayBasisResult(container, data) { ... original code ... } */
}); 
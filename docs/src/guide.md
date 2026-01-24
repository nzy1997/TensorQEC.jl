# Getting Started

This guide covers the three primary workflows of TensorQEC v0.3:
1. **QEC Simulation** -- define a code, simulate errors, and decode
2. **Code Analysis** -- define a code and compute its properties
3. **STIM Interop** -- parse STIM circuit files, analyze, and decode

## 1. QEC Simulation

The simulation workflow lets you evaluate decoder performance under a noise model. The typical steps are: construct a code, define an error model, build a decoding problem, compile a decoder, then run Monte Carlo trials.

### Full Example

```julia
using TensorQEC
using Random

# 1. Construct a quantum error-correcting code
code = SteaneCode()          # [[7,1,3]] Steane code
# Other options: SurfaceCode(3,3), ToricCode(3,3), ShorCode(), Code513(), etc.

# 2. Create an i.i.d. depolarizing error model (px, py, pz, n)
n = code_n(code)             # number of physical qubits
em = iid_error(0.05, 0.05, 0.05, n)

# 3. Build a DecodingProblem (wraps code + error model)
problem = DecodingProblem(code, em)

# 4. Choose a decoder and compile it for this problem
decoder = BPDecoder()        # Belief Propagation + OSD
compiled = compile(decoder, problem)
# Other decoders: IPDecoder(), MatchingDecoder(), TNMAP(), TNMMAP(), TableDecoder()

# 5. Build auxiliary structures for error checking
tanner = CSSTannerGraph(code)
lx, lz = logical_operator(tanner)

# 6. Monte Carlo simulation loop
num_trials = 1000
logical_errors = 0

for _ in 1:num_trials
    # Generate a random error
    error = random_error_pattern(em)

    # Extract the syndrome
    syndrome = syndrome_extraction(error, tanner)

    # Decode
    result = decode(compiled, syndrome)

    # Check for logical error
    if result.success_tag
        if check_logical_error(error, result.error_pattern, lx, lz)
            logical_errors += 1
        end
    else
        logical_errors += 1  # decoding failure counts as logical error
    end
end

logical_error_rate = logical_errors / num_trials
println("Logical error rate: $logical_error_rate")
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `iid_error(px, py, pz, n)` | Create an i.i.d. depolarizing error model |
| `DecodingProblem(code, em)` | Bundle code and error model into a decoding problem |
| `compile(decoder, problem)` | Pre-compile a decoder (amortizes setup cost) |
| `random_error_pattern(em)` | Sample a random error from the error model |
| `syndrome_extraction(error, tanner)` | Compute the syndrome for a given error |
| `decode(compiled, syndrome)` | Decode a syndrome; returns a `DecodingResult` |
| `check_logical_error(e1, e2, lx, lz)` | Check if the difference of two errors is a logical operator |

### Available Decoders

| Decoder | Type | Notes |
|---------|------|-------|
| `BPDecoder()` | Classical | Belief Propagation with OSD fallback |
| `MatchingDecoder(solver)` | Classical | Minimum-weight perfect matching (solvers: `TensorQEC.IPMatchingSolver()`, `TensorQEC.GreedyMatchingSolver()`) |
| `IPDecoder()` | General | Integer programming (exact, slower) |
| `TNMAP()` | General | Tensor network MAP decoder |
| `TNMMAP()` | General | Tensor network marginal MAP decoder |
| `TableDecoder(d)` | General | Lookup table (d = code distance, small codes only) |

Classical decoders handle CSS codes by decomposing into independent X/Z decoding. General decoders handle arbitrary error correlations.

## 2. Code Analysis

Use this workflow to inspect the structure and parameters of a quantum code.

### Full Example

```julia
using TensorQEC

# 1. Construct a code
code = SurfaceCode(3, 3)     # 3x3 rotated surface code

# 2. Get the stabilizer generators
stabs = stabilizers(code)
println("Number of stabilizers: ", length(stabs))
println("First stabilizer: ", stabs[1])

# 3. Compute code parameters [n, k, d]
n = code_n(code)    # physical qubits
s = code_s(code)    # number of stabilizers
k = code_k(code)    # logical qubits (n - s for these codes)
println("[[n, k]] = [[$n, $k]]")

# 4. Build a CSS Tanner graph
tanner = CSSTannerGraph(code)

# 5. Compute the code distance (uses integer programming)
d = code_distance(tanner)
println("Code distance: $d")

# 6. Find logical operators
lx, lz = logical_operator(tanner)
println("Number of logical X operators: ", size(lx, 1))
println("Number of logical Z operators: ", size(lz, 1))
```

### Available Codes

| Code | Constructor | Parameters |
|------|-------------|------------|
| Steane code | `SteaneCode()` | [[7,1,3]] |
| Surface code | `SurfaceCode(m, n)` | Rotated, m-by-n |
| Toric code | `ToricCode(m, n)` | m-by-n on torus |
| Shor code | `ShorCode()` | [[9,1,3]] |
| 5-qubit code | `Code513()` | [[5,1,3]] |
| [[4,2,2]] code | `Code422()` | [[4,2,2]] |
| [[8,3,2]] code | `Code832()` | [[8,3,2]] |
| [[15,7,3]] code | `Code1573()` | [[15,7,3]] |
| Bivariate bicycle | `BivariateBicycleCode(m, n, vc, hd)` | General LDPC |
| Color 4.8.8 | `Color488(d)` | Distance d |
| Color 6.6.6 | `Color666(d)` | Distance d |

### Key Functions

| Function | Purpose |
|----------|---------|
| `stabilizers(code)` | Get stabilizer generators as `PauliString` vectors |
| `code_n(code)` | Number of physical qubits |
| `code_s(code)` | Number of stabilizer generators |
| `code_k(code)` | Number of logical qubits |
| `CSSTannerGraph(code)` | Build CSS Tanner graph from a code |
| `code_distance(tanner)` | Compute minimum distance via integer programming |
| `logical_operator(tanner)` | Compute logical X and Z operator matrices |

## 3. STIM Interop

TensorQEC can parse [STIM](https://github.com/quantumlib/Stim) circuit files and extract detector error models, enabling integration with STIM-based workflows.

### Parsing a STIM File

```julia
using TensorQEC

# Parse a STIM circuit file into a Yao circuit
qc = parse_stim_file("path/to/circuit.stim", 7)
# The second argument is the number of qubits in the circuit
```

The returned `ChainBlock` is a Yao circuit containing gates, measurements, detectors, and noise channels from the STIM file.

### Extracting a Detector Error Model

Once you have a parsed circuit, you can extract a detector error model (DEM) that describes how each error channel affects the detectors.

```julia
using TensorQEC

# Parse a noisy circuit
qc = parse_stim_file("noisy_repetition.stim", 7)

# Extract the detector error model
dem = detector_error_model(qc)

# The DEM contains:
# - dem.error_rates: probability of each error mechanism
# - dem.flipped_detectors: which detectors each error flips
# - dem.detector_list: list of detector indices
# - dem.logical_list: list of logical observable indices
```

### Decoding from a DEM

You can convert a detector error model into a Tanner graph and use TensorQEC's decoders.

```julia
using TensorQEC

# Parse and extract DEM
qc = parse_stim_file("noisy_repetition.stim", 7)
dem = detector_error_model(qc)

# Convert DEM to a Tanner graph for decoding
tanner = TensorQEC.dem2tanner(dem)

# Create an error model from the DEM error rates
pvec = IndependentFlipError(dem.error_rates)

# Compile and decode using the Tanner graph and error model
compiled = compile(BPDecoder(), tanner, pvec)

# Generate a random error pattern from the DEM
error = random_error_pattern(dem)

# Extract syndrome (only detector bits, not logical bits)
syndrome = SimpleSyndrome(tanner.H * error)

# Decode
result = decode(compiled, syndrome)
```

### Inserting Noise into a Circuit

You can also add noise channels to a parsed STIM circuit programmatically.

```julia
using TensorQEC

# Parse a clean circuit
qc = parse_stim_file("circuit.stim", 7)

# Insert noise: depolarizing after Cliffords, bit-flip after reset, bit-flip before measure
noisy_qc = TensorQEC.insert_errors(qc;
    after_clifford_depolarization=0.001,
    after_reset_flip_probability=0.001,
    before_measure_flip_probability=0.001
)

# Now extract DEM from the noisy circuit
dem = detector_error_model(noisy_qc)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `parse_stim_file(path, nqubits)` | Parse a STIM file into a Yao circuit |
| `detector_error_model(qc)` | Extract detector error model from a circuit |
| `TensorQEC.dem2tanner(dem)` | Convert DEM to a `SimpleTannerGraph` for decoding |
| `random_error_pattern(dem)` | Sample a random error from the DEM |
| `TensorQEC.insert_errors(qc; ...)` | Add noise channels to a circuit |

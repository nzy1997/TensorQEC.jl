# API Reference

## Codes

```@docs
SurfaceCode
ToricCode
ShorCode
SteaneCode
Code832
Code422
Code1573
Code513
BivariateBicycleCode
Color488
Color666
stabilizers
code_distance
logical_operator
```

## Pauli Algebra

```@docs
Pauli
PauliString
PauliGroupElement
pauli_decomposition
pauli_basis
pauli_repr
isanticommute
yaoblock
SumOfPaulis
```

## Clifford Simulation

```@docs
CliffordGate
clifford_simulate
compile_clifford_circuit
Tableau
tableau_simulate
```

## Decoding Pipeline

```@docs
DecodingProblem
compile
decode
DecodingResult
BPDecoder
IPDecoder
MatchingDecoder
TableDecoder
TNMAP
TNMMAP
```

## Error Models

```@docs
IndependentFlipError
IndependentDepolarizingError
iid_error
random_error_pattern
CSSErrorPattern
SimpleSyndrome
CSSSyndrome
syndrome_extraction
check_logical_error
```

## Tanner Graphs

```@docs
SimpleTannerGraph
CSSTannerGraph
product_graph
random_ldpc
```

## Encoding

```@docs
encode_stabilizers
place_qubits
```

## Measurement & DEM

```@docs
measurement_circuit
measure_circuit_fault_tol
DetectorErrorModel
detector_error_model
```

## Threshold

```@docs
multi_round_qec
```

## STIM Interop

```@docs
parse_stim_file
```

## Tensor Network Simulation

```@docs
qc2einsum
coherent_error_unitary
fidelity_tensornetwork
simulation_tensornetwork
ComplexConj
QCInfo
```

## Utilities

```@docs
Mod2
```

## Multiprocessing

```@docs
TensorQEC.SimpleMultiprocessing.multiprocess_run
```

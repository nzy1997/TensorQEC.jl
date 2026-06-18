# Logical Clifford Action Design

Date: 2026-06-05

## Goal

Add a reusable verification framework that computes how a physical Clifford action induces a logical Clifford action on a stabilizer code.

The immediate acceptance target is the transversal CNOT between two `ToricCode(3,3)` blocks, verified entirely at the Pauli/matrix level.

## Scope

In scope:

- Reducing a physical Pauli operator to logical Pauli coordinates modulo stabilizers.
- Computing the logical action induced by an arbitrary physical Clifford action on the logical Pauli generators.
- Verifying that the physical Clifford preserves the code space.
- Supporting stabilizer codes described by `stabilizers`, `lx`, and `lz`.
- Using the double-`ToricCode(3,3)` transversal CNOT as the primary end-to-end test.

Out of scope for the first version:

- Automatically naming the induced logical Clifford as `CNOT`, `H`, `S`, etc.
- Synthesizing a logical circuit from the induced action.
- Handling non-Clifford physical operations.
- Building a fully general symplectic linear algebra framework beyond the needs of this API.

## Design Summary

The implementation is split into two layers:

1. A logical-coordinate reducer for physical Pauli operators.
2. A logical-Clifford analyzer that applies a user-supplied physical Clifford action to logical generators and reduces the resulting Pauli images.

This separation keeps the reusable algebraic primitive small and makes the Clifford layer a thin orchestration step.

## Existing Building Blocks

The repository already provides the core ingredients needed for this design:

- `PauliString` and `PauliGroupElement` for physical Pauli operators.
- `iscommute` / `isanticommute` for Pauli commutation checks.
- `logical_operator(tanner)` to derive logical X/Z bases.
- `same_qubit_order` to align the logical X/Z bases.
- `CliffordGate` application on `PauliString` / `PauliGroupElement` for Clifford conjugation.

No state simulation, encoder simulation, or new circuit abstraction is required.

## API

### Logical Pauli Reduction

Add:

```julia
logical_pauli_coordinates(
    stabilizers::AbstractVector{PauliString{N}},
    lx::AbstractMatrix{Mod2},
    lz::AbstractMatrix{Mod2},
    op::PauliString{N},
) where N
```

Return:

```julia
struct LogicalPauliCoordinates
    preserves_stabilizers::Bool
    x_bits::Vector{Bool}
    z_bits::Vector{Bool}
end
```

Semantics:

- If `preserves_stabilizers == true`, then `op` is logically equivalent to
  `prod_j X_j^(x_bits[j]) Z_j^(z_bits[j])`,
  up to multiplication by stabilizers and an overall phase.
- If `preserves_stabilizers == false`, the bit vectors are still diagnostic but should not be treated as a valid logical Pauli representative.

### Logical Clifford Analysis

Add:

```julia
logical_clifford_action(
    stabilizers::AbstractVector{PauliString{N}},
    lx::AbstractMatrix{Mod2},
    lz::AbstractMatrix{Mod2},
    act_on_pauli::Function,
) where N
```

Where `act_on_pauli(op)` returns the conjugated Pauli `U op U†` as a `PauliGroupElement`.

Return:

```julia
struct LogicalCliffordAction
    preserves_code::Bool
    stabilizer_images::Vector{LogicalPauliCoordinates}
    x_images::Vector{LogicalPauliCoordinates}
    z_images::Vector{LogicalPauliCoordinates}
end
```

Semantics:

- `stabilizer_images[i]` is the reduced image of the `i`th stabilizer generator.
- `x_images[j]` is the reduced image of logical `X_j`.
- `z_images[j]` is the reduced image of logical `Z_j`.
- `preserves_code == true` means the supplied physical Clifford preserves the encoded subspace and induces a well-defined logical Clifford action on the provided logical basis.

## Data Representation

`lx` and `lz` are binary `Mod2` matrices with one row per logical qubit. Each row is converted into a physical `PauliString` using the same conventions already used in `verify_logical_action`:

- a row from `lx` becomes an X-type `PauliString`
- a row from `lz` becomes a Z-type `PauliString`

For `k` logical qubits, the logical Pauli coordinates are represented by two length-`k` Boolean vectors:

- `x_bits[j] == true` means the logical image contains `X_j`
- `z_bits[j] == true` means the logical image contains `Z_j`

This representation naturally supports:

- pure X-type or Z-type logical operators
- mixed logical Paulis containing both X and Z support
- logical Clifford actions that interchange X and Z structure

## Logical Pauli Reduction Semantics

For a physical Pauli operator `op`, reduction proceeds as follows:

1. Check commutation with every stabilizer generator.
2. If any stabilizer anticommutes with `op`, set `preserves_stabilizers = false`.
3. Compute logical coordinates by commutation against the logical basis:
   - `x_bits[j] = isanticommute(op, logical_z_j)`
   - `z_bits[j] = isanticommute(op, logical_x_j)`

This works because the provided `lx`/`lz` basis is already paired by `same_qubit_order`, so each pair forms a canonical logical symplectic basis.

The reducer does not need to explicitly solve for the stabilizer multiplier. The commutation signature already identifies the logical Pauli class modulo stabilizers.

## Code-Preservation Semantics

For a physical Clifford action `U`, `preserves_code` should be `true` exactly when all of the following hold:

1. For every stabilizer generator `s`, the image `U s U†` preserves stabilizers.
2. For every stabilizer generator image, the reduced logical coordinates are trivial:
   - all `x_bits` are `false`
   - all `z_bits` are `false`
3. For every logical generator image `U X_j U†` and `U Z_j U†`, the image preserves stabilizers.

This is stricter than merely asking stabilizer images to commute with the stabilizer group. A stabilizer image with nontrivial logical coordinates would not preserve the code space pointwise and must therefore be rejected.

## Physical Clifford Interface

The first version should accept the physical action as a function:

```julia
act_on_pauli(op::PauliString{N}) -> PauliGroupElement{N}
```

This keeps the API compatible with existing repository patterns:

- a transversal CNOT can be implemented by repeatedly applying `CliffordGate(ConstGate.CNOT)`
- a compiled Clifford circuit can later be wrapped behind the same callable shape

The analyzer should treat phase as irrelevant for logical-coordinate extraction, while preserving the `PauliGroupElement` return type to stay faithful to the underlying Clifford action.

## Validation and Error Handling

The new APIs should assert:

- `size(lx, 1) == size(lz, 1)`
- `size(lx, 2) == size(lz, 2) == N`
- every stabilizer has length `N`

The APIs trust the caller that:

- the supplied stabilizers define a valid stabilizer code
- the supplied `lx` and `lz` define a valid paired logical basis for that code

The APIs should not attempt to reconstruct or repair an invalid logical basis.

## File Placement

Implementation should live next to the existing logical-operator utilities:

- add the new result types and functions in `src/codes/code_distance.jl`
- export them from `src/TensorQEC.jl`

Tests should live in:

- `test/codes/code_distance.jl`

This keeps logical-operator diagnostics and logical-action verification in one place.

## Testing

### 1. Logical Pauli Reduction

Add focused tests that confirm known logical operators reduce to the expected coordinates:

- `SteaneCode()` single-logical-qubit checks
- `ToricCode(3,3)` multi-logical-qubit checks

Expected behavior:

- a physical representative of logical `X_j` yields `x_bits[j] == true` and all other logical bits `false`
- a physical representative of logical `Z_j` yields `z_bits[j] == true` and all other logical bits `false`
- a non-logical Pauli that breaks stabilizers sets `preserves_stabilizers == false`

### 2. Logical Clifford Framework

Add a small test where the physical action is known and easy to reason about, using a callable `act_on_pauli` and checking that `logical_clifford_action` returns the expected logical-generator images.

The test should verify both:

- the returned logical coordinates
- the `preserves_code` decision

### 3. Acceptance Test: Double Toric Transversal CNOT

Construct two `ToricCode(3,3)` blocks as an 18-qubit system:

1. Build one block's stabilizers and logical basis.
2. Embed one copy as control qubits `1:9` and one copy as target qubits `10:18`.
3. Form the combined stabilizer set and combined logical basis.
4. Define `act_on_pauli` by composing the 9 physical pairwise CNOT conjugations.

The expected logical action is two independent logical CNOTs, one per logical pair:

- `Xc_1 -> Xc_1 Xt_1`
- `Xc_2 -> Xc_2 Xt_2`
- `Xt_1 -> Xt_1`
- `Xt_2 -> Xt_2`
- `Zc_1 -> Zc_1`
- `Zc_2 -> Zc_2`
- `Zt_1 -> Zc_1 Zt_1`
- `Zt_2 -> Zc_2 Zt_2`

This test is the first required proof that the framework can answer the user's original verification question without building a state-level circuit simulation.

## Non-Goals

The first implementation should avoid:

- introducing a new global abstraction for arbitrary symplectic maps
- inferring named logical gates from the returned Pauli-generator images
- expanding into non-CSS-specific convenience helpers unless needed by the tests

The goal is a compact, defensible algebraic API with one real acceptance case.

# Logical Action Verification Design

Date: 2026-06-04

## Goal

Add a small verification utility for CSS codes that checks whether a physical Pauli operator preserves the stabilizer group and how it commutes with the code's logical X/Z operators.

This utility is meant for code-level diagnostics and tests. The first target test is the Steane code logical X operator.

## Scope

In scope:

- Verify stabilizer preservation for a physical Pauli operator.
- Report commutation against each provided logical X operator.
- Report commutation against each provided logical Z operator.
- Support general CSS codes with `k >= 1` logical qubits.

Out of scope:

- Identifying a final logical label such as logical `I/X/Y/Z`.
- Verifying non-Pauli Clifford gates.
- Building the verification around state simulation or encoding-circuit simulation.

## Chosen Approach

Use pure Pauli algebra on existing `PauliString` values.

For a Pauli operator `op`, stabilizer preservation is checked by testing commutation with every stabilizer generator in `stabilizers`. Logical action is diagnosed by testing commutation with every row-derived logical operator from `lx` and `lz`.

This approach matches the existing codebase:

- `PauliString` already represents physical Pauli operators.
- `iscommute` / `isanticommute` already implement Pauli commutation checks.
- `logical_operator(tanner)` already produces logical X/Z operators as `Mod2` matrices.

No new simulation layer is needed.

## API

Add:

```julia
verify_logical_action(
    stabilizers::AbstractVector{PauliString{N}},
    lx::AbstractMatrix{Mod2},
    lz::AbstractMatrix{Mod2},
    op::PauliString{N},
) where N
```

The function will return a small result object:

```julia
struct LogicalActionVerification
    preserves_stabilizers::Bool
    commutes_with_stabilizers::Vector{Bool}
    commutes_with_lx::Vector{Bool}
    commutes_with_lz::Vector{Bool}
end
```

This keeps the interface explicit and easy to assert against in tests.

## Data Conversion

`lx` and `lz` are provided as binary matrices over `Mod2`, one row per logical operator.

Each logical row will be converted to a `PauliString` using existing repository conventions:

- a row from `lx` becomes an X-type `PauliString`
- a row from `lz` becomes a Z-type `PauliString`
- active qubit positions are extracted with `findall(i -> i.x, row)`

This matches existing patterns already used elsewhere in the repository.

## Semantics

### Stabilizer Preservation

For the current scope, `op` is always a Pauli operator. A Pauli operator preserves the stabilizer group iff it commutes with every stabilizer generator supplied in `stabilizers`.

The function will therefore compute:

- `commutes_with_stabilizers[i] = iscommute(op, stabilizers[i])`
- `preserves_stabilizers = all(commutes_with_stabilizers)`

### Logical Commutation Diagnostics

For each logical operator row:

- `commutes_with_lx[j] = iscommute(op, logical_x_j)`
- `commutes_with_lz[j] = iscommute(op, logical_z_j)`

For `k = 1`, this reduces to one Bool for logical X and one Bool for logical Z.

For `k > 1`, the result reports one Bool per logical qubit, preserving the row order already established by `logical_operator`.

## Validation and Error Handling

The function should defensively assert:

- `size(lx, 1) == size(lz, 1)` so both logical bases describe the same number of logical qubits.
- every stabilizer and `op` act on the same number of qubits.
- `size(lx, 2)` and `size(lz, 2)` match that qubit count.

No attempt will be made to re-derive or validate that the supplied `lx` and `lz` are a correct logical basis; the function trusts the caller on that point.

## Testing

Add a focused testset using `SteaneCode()`:

1. Build `st = stabilizers(SteaneCode())`.
2. Build `tanner = CSSTannerGraph(st)`.
3. Compute `lx, lz = logical_operator(tanner)`.
4. Convert `lx[1, :]` into a physical X-type `PauliString`.
5. Run `verify_logical_action(st, lx, lz, op)`.

Expected assertions:

- `result.preserves_stabilizers == true`
- `all(result.commutes_with_stabilizers)`
- `result.commutes_with_lx == [true]`
- `result.commutes_with_lz == [false]`

This verifies that the chosen physical operator behaves as Steane logical X: it preserves the code space, commutes with logical X, and anticommutes with logical Z.

## File Placement

Implementation should stay close to the existing logical-operator utilities. The preferred placement is alongside code-distance and logical-operator helpers, with exports added from `src/TensorQEC.jl`.

Tests should live with the existing logical-operator tests under `test/codes/code_distance.jl`, unless a cleaner nearby file becomes obvious during implementation.

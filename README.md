<p align="center">
<img width="400" src="./docs/src/images/logoname.svg"/>
</p>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nzy1997.github.io/TensorQEC.jl/dev/)
[![CI](https://github.com/nzy1997/TensorQEC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/nzy1997/TensorQEC.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/nzy1997/TensorQEC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nzy1997/TensorQEC.jl)

This package utilizes the tensor network to study the properties of *quantum error correction* (QEC).The main features include
* A collection of QEC codes, their stabilizer generators, and their encoding circuits,
* Code distance calculation with integer programming,
* Decoders: truth table, tensor network, integer programming, BPOSD, etc.
* Simulation backends: tensor network, clifford circuit, full amplitude simulation (with Yao.jl), etc.

## Installation

TensorQEC is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install TensorQEC,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press the <kbd>]</kbd> key in the REPL to use the package mode, and then type:
</p>

```julia
pkg> add TensorQEC
```

To update, just type `up` in the package mode.

## Contribute

Suggestions and Comments in the [_Issues_](https://github.com/nzy1997/TensorQEC.jl/issues) are welcome.

Contributions to the documentation are welcome. To build the documentation, please run:
```
make init init-docs  # or make update-docs
make servedocs
```

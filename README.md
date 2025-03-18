<p align="center">
<img width="400" src="./docs/src/images/logoname.svg"/>
</p>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nzy1997.github.io/TensorQEC.jl/dev/)
[![CI](https://github.com/nzy1997/TensorQEC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/nzy1997/TensorQEC.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/nzy1997/TensorQEC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nzy1997/TensorQEC.jl)

This package utilizes the tensor network to study the properties of *quantum error correction* (QEC).The main features include
* Quantum error correction code decoder with tensor network,
* Quantum circuit simulation with tensor network to estimate the threshold of QEC.

Also, we include more general QEC tools, including
* Commonly used QEC code stabilizer generators,
* QEC code distance calculation,
* QEC encoding circuit construction,
* Decoding truth table construction,
* Measurement circuit construction.

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
pkg> add https://github.com/nzy1997/TensorQEC.jl.git
```

To update, just type `up` in the package mode.

## Contribute

Suggestions and Comments in the [_Issues_](https://github.com/nzy1997/TensorQEC.jl/issues) are welcome.

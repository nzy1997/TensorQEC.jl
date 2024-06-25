# TensorQEC

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nzy1997.github.io/TensorQEC.jl/stable/)
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

Type `]` in a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/index.html) to enter the `pkg` mode, then type
```julia pkg
pkg> dev https://github.com/nzy1997/TensorQEC.jl.git
```

## Contribute

Suggestions and Comments in the _Issues_ are welcome.

## License
MIT License
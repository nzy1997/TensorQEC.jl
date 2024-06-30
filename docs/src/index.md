```@meta
CurrentModule = TensorQEC
```

# TensorQEC

This package utilizes the tensor network to study the properties of *quantum error correction*(QEC). The main features include
* Quantum error correction decoder with tensor network 
  Essentially, quantum error inference is a probabilistic graphical model problem. With the help of package `TensorInference.jl`, we can use the tensor network to solve it. However, to get the accurate result, we need to maximize a posteriori probability with an exponentially large number of possible configurations. Here we only maximize the marginal probability of each qubit to get an estimation. It works well for small systems. For more details, please check section [Inference with Tensor Network](@ref).
* Non-Clifford quantum circuit simulation with tensor network
  Most quantum gates in QEC protocols are Clifford gate, which can be simulated efficiently. To simulate non-Clifford gates, we convert quantum circuit to tensor networks. Section [Measurement-Free QEC](@ref) and [Coherent Error Simulation](@ref) are two examples of simulating non-Clifford gates. However, for a deep circuit, the cost for tensor networks simulating quantum circuits is also exponential.

Also, we include more general QEC tools, including
* Commonly used QEC code stabilizer generators,
* QEC encoding circuit construction,
* Decoding truth table construction,
* Measurement circuit construction,
* Clifford gate simulation.
  
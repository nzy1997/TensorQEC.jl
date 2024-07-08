```@meta
CurrentModule = TensorQEC
```

# TensorQEC

This package utilizes the tensor network to study the properties of *quantum error correction*(QEC). The main features include
* *Incoherent Quantum error correction:*
  In the incoherent quantum error correction scheme, finding the most likely true error from the error syndrome is a standard probabilistic inference problem on boolean variables.
  This problem is closely connected to tensor networks[^Ferris], which can be solved with existing tools such as [`TensorInference.jl`](https://github.com/TensorBFS/TensorInference.jl). Examples are given in the following sections:
  * [Inference with Tensor Network](@ref)
* *Coherent quantum error correction:*
  Unlike Clifford gate, non-Clifford gates can not be simulated efficiently in general. By converting the quantum circuit into a tensor network, we can simulate small coherent quantum error correction circuits. Examples are given in the following sections:
    * [Coherent Error Simulation](@ref)
    * [Measurement-Free QEC](@ref)
  
[^Ferris]: Ferris, A. J.; Poulin, D. Tensor Networks and Quantum Error Correction. Phys. Rev. Lett. 2014, 113 (3), 030501. https://doi.org/10.1103/PhysRevLett.113.030501.
# # Mixed-Integer Programming for Decoding
# ## Problem Statement
# The parity-check matrices of a CSS quantum code are $H_x \in \mathbb{F}^{m_x \times n}_2$ and $H_z \in \mathbb{F}^{m_z \times n}_2$ where $\mathbb{F}_2$ is the finite field with two elements, $n$ is the number of qubits, $m_x$ is the number of $X$-stabilizers, and $m_z$ is the number of $Z$-stabilizers. We can use [`CSSTannerGraph`](@ref) to generate a tanner graph for a CSS quantum code.
using TensorQEC
tanner = CSSTannerGraph(SteaneCode());
# And the parity-check matrix of $X$-stabilizers of the Steane code is
tanner.stgx.H

# The error vectors $\mathbf{x,y,z} \in \mathbb{F}^n_2$ are binary vectors. The $j$-th element of $\mathbf{x}$ is $1$ if the $j$-th qubit is flipped by an $X$-error, and $0$ otherwise. There is at most one error per qubit, i.e., $\mathbf{x}_j + \mathbf{y}_j + \mathbf{z}_j \leq 1$. [`random_error_pattern`](@ref) can be used to generate a random error pattern for a given number of qubits and an error model.
using Random; Random.seed!(110)
error_pattern = random_error_pattern(iid_error(0.1,0.1,0.1,7))
# Here we decompose $Y$ errors into $X$ and $Z$ errors. The error pattern is $Y_4X_6 =iX_4X_6Z_4$.

# The syndrome of $X$-stabilizers and $Z$-stabilizers are $H_x(\mathbf{y}+\mathbf{z}) = s_x \in \mathbb{F}^{m_x}_2$ and $H_z (\mathbf{x}+\mathbf{y}) = s_z \in \mathbb{F}^{m_z}_2$. We can use [`syndrome_extraction`](@ref) to extract the syndrome of a given error pattern.
syndrome = syndrome_extraction(error_pattern,tanner)
# The goal is to find the most-likely error $\mathbf{x},\mathbf{y},\mathbf{z} \in \mathbb{F}^n_2$ given the syndrome $s_x$ and $s_z$.

# Suppose that the error distributions on different qubits are independent to each other. And we use $p_{\sigma j}$ to denote the probability of the $j$-th qubit being flipped by an error of type $\sigma \in \{x,y,z\}$. Then the logarithm of the total error probability is 
# ```math
# L(\mathbf{x,y,z}) = \sum_{j=1}^n (\mathbf{x}_j \log p_{xj} + \mathbf{y}_j \log p_{yj} + \mathbf{z}_j \log p_{zj} + (1-\mathbf{x}_j-\mathbf{y}_j-\mathbf{z}_j) \log (1-p_{xj}-p_{yj}-p_{zj})).
# ```
# The resulting mixed-integer program can be summarized as:
# ```math
# \begin{aligned}
# \text{maximize} \quad & L(\mathbf{x,y,z}) \\
# \text{subject to} \quad & H_x (\mathbf{y+z}) = s_x, \\
# & H_z (\mathbf{x+y}) = s_z, \\
# & \mathbf{x,y,z} \in \{0,1\}^n, \\
# & \mathbf{x}_j + \mathbf{y}_j + \mathbf{z}_j \leq 1, \quad j=1,\ldots,n.
# \end{aligned}
# ```

# ## Mixed-Integer Programming
# Since $H_x (\mathbf{y+z}) = s_x$ in $\mathbb{F}_2$ is equivalent to $H_x (\mathbf{y+z}) = s_x \mod 2$ in $\mathbb{Z}$, we can convert above programming problem into a mixed-integer programming problem as follows:
# ```math
# \begin{aligned}
# \text{maximize} \quad & L(\mathbf{x,y,z}) \\
# \text{subject to} \quad & H_x (\mathbf{y+z}) = s_x + 2\mathbf{k}, \\
# & H_z (\mathbf{x+y}) = s_z + 2\mathbf{l}, \\
# & \mathbf{x,y,z} \in \{0,1\}^n, \\
# & \mathbf{x}_j + \mathbf{y}_j + \mathbf{z}_j \leq 1, \quad j=1,\ldots,n.\\
# & \mathbf{k} \in \mathbb{Z}^{m_x}, \mathbf{l} \in \mathbb{Z}^{m_z}.
# \end{aligned}
# ```
# Here $\mathbf{k}$ and $\mathbf{l}$ are auxiliary variables to convert the modulo operation into linear constraints.

# We implement the above mixed-integer programming problem in [`IPDecoder`](@ref) and solve it with [JuMP.jl](https://github.com/jump-dev/JuMP.jl) and [HiGHS.jl](https://github.com/jump-dev/HiGHS.jl). We can use [`decode`](@ref) to decode the syndrome with this integer programming decoder.
decoder = IPDecoder()
decode(decoder, tanner, syndrome)

# Here we get a different error pattern $X_2Z_4$. That is because the default error probability is $0.05$ for each qubit and each error type. And this error pattern has the same syndrome as the previous one. If we slightly increase the $X$ and $Y$ error probability, we can get the correct error pattern $Y_4X_6$.
decode(decoder, tanner, syndrome, iid_error(0.06, 0.06, 0.05,7))
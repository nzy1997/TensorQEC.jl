# Performance Tips
## Multiprocessing
Submodule `TensorQEC.SimpleMutiprocessing` provides one function [`TensorQEC.SimpleMultiprocessing.multiprocess_run`](@ref) function for simple multi-processing jobs.
It is not directly related to `TensorQEC`, but is very convenient to have one.
Suppose we want to decode $9 \times 9$ surface code for 10 different error syndromes with 4 processes.
We can create a file, e.g. named `run.jl` with the following content

```julia
using Distributed, TensorQEC.SimpleMultiprocessing
using Random, TensorQEC  # to avoid multi-precompilation
@everywhere using Random, TensorQEC

results = multiprocess_run(collect(1:10)) do seed
    Random.seed!(seed)
    d = 9
    @info "$d x $d SurfaceCode, error seed= $seed"
    decoder = IPDecoder()
    tanner = CSSTannerGraph(SurfaceCode(d, d))
    em = DepolarizingError(0.05, 0.06, 0.1)
    ep = random_error_pattern(d*d, em)
    syn = syndrome_extraction(ep,tanner)
    res = decode(decoder,tanner,syn)
    return check_logical_error(ep.xerror, res.xerror_pattern, tanner.stgx.H) && check_logical_error(ep.zerror, res.zerror_pattern, tanner.stgz.H)
end

println(results)
```

One can run this script file with the following command
```bash
$ julia -p4 run.jl
      From worker 4:    [ Info: running argument 4 on device 4
      From worker 2:    [ Info: running argument 3 on device 2
      From worker 2:    [ Info: 9 x 9 SurfaceCode, error seed= 3
      From worker 3:    [ Info: running argument 1 on device 3
      From worker 3:    [ Info: 9 x 9 SurfaceCode, error seed= 1
      From worker 4:    [ Info: 9 x 9 SurfaceCode, error seed= 4
      From worker 5:    [ Info: running argument 2 on device 5
      From worker 5:    [ Info: 9 x 9 SurfaceCode, error seed= 2
      From worker 3:    [ Info: running argument 5 on device 3
      From worker 3:    [ Info: 9 x 9 SurfaceCode, error seed= 5
      ...
    Any[false, true, true, true, false, true, true, true, false, false]
```
You will see a vector of decoding results printed out, where `true` means there is no logical error according to the decoding result.
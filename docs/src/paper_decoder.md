# Paper Decoder Snapshot

The `paper/topological-css-decoder` branch preserves the historical
unitary-decouple decoder used for the topological CSS code decoupling article.
The branch is archival: it exists to reproduce the article's decoder workflow
and is not the current recommended TensorQEC decoder API.

Run the deterministic color-code fixture smoke check from the repository root:

```bash
julia --project examples/paper_decoder_smoke.jl
```

The decoder maps the transformed parity checks to toric-code matching
subproblems. Calls to SparseBlossom add edges without weights, so the MWPM
instances are unweighted. Packed binary matrix multiplication is part of the
decoder kernel preserved by this snapshot.

The article's decoder timing measures only the `decode` call after compilation.
It excludes error sampling, syndrome generation, syndrome validation, decoder
compilation, process startup, and file I/O. The companion DecoderBenchmarks
release records the warm-up count, sample count, machine environment, and raw
timing provenance.

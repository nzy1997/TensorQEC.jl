using TensorQEC
using Documenter
using Literate
using DocThemeIndigo

for each in readdir(pkgdir(TensorQEC, "examples"))
    input_file = pkgdir(TensorQEC, "examples", each)
    endswith(input_file, ".jl") || continue
    @info "building" input_file
    output_dir = pkgdir(TensorQEC, "docs", "src", "generated")
    Literate.markdown(input_file, output_dir; name=each[1:end-3], execute=false)
end

indigo = DocThemeIndigo.install(TensorQEC)
DocMeta.setdocmeta!(TensorQEC, :DocTestSetup, :(using TensorQEC); recursive=true)

makedocs(;
    modules=[TensorQEC],
    authors="Zhongyi Ni",
    sitename="TensorQEC.jl",
    format=Documenter.HTML(;
        canonical="https://nzy1997.github.io/TensorQEC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Topics" => [
            "QEC Codes" => "generated/codes.md",
            "Pauli Basis and Clifford group" => "generated/clifford.md",
            "Inference with Tensor Network" => "generated/inference.md",
            "Coherent Error Simulation" => "generated/coherent.md",
            "Measurement-Free QEC" => "generated/simulation.md",
            "Decoders" => [
                "Mixed-Integer Programming Decoder" => "generated/ipdecoder.md",
                # MORE
            ],
            "Performance Tips" => "performancetips.md",
        ],
        "Manual" => "man.md",
    ],
)

deploydocs(;
    repo="github.com/nzy1997/TensorQEC.jl",
    devbranch="main",
    push_preview=true,
)

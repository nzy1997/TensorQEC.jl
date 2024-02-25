using TensorQEC
using Documenter

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
    ],
)

deploydocs(;
    repo="github.com/nzy1997/TensorQEC.jl",
    devbranch="main",
)

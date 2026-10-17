using CFBoxes
using Documenter

DocMeta.setdocmeta!(CFBoxes, :DocTestSetup, :(using CFBoxes); recursive=true)

makedocs(;
    modules=[CFBoxes, CFBoxes.CGrid],
    authors="Thomas Dubos <thomas.dubos@polytechnique.edu> and contributors",
    sitename="CFBoxes.jl",
    format=Documenter.HTML(;
        canonical="https://dubosipsl.github.io/CFBoxes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dubosipsl/CFBoxes.jl",
    devbranch="main",
)

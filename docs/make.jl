using SegmentationModels
using Documenter

makedocs(;
    modules=[SegmentationModels],
    authors="Max Freudenberg",
    repo="https://github.com/maxfreu/SegmentationModels.jl/blob/{commit}{path}#L{line}",
    sitename="SegmentationModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://maxfreu.github.io/SegmentationModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maxfreu/SegmentationModels.jl",
)

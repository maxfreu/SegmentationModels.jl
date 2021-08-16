module SegmentationModels

using Reexport
using Flux
@reexport using Metalhead

const Classifier = Union{ResNet, VGG}

include("encoders.jl")
include("unet.jl")

export UNet

end

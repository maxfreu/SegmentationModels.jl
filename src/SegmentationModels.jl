module SegmentationModels

using Reexport
using Flux
@reexport using Metalhead

const Classifier = Union{ResNet, ResNeXt, VGG, MobileNetv1, MobileNetv2, MobileNetv3}

include("utils.jl")
include("encoders.jl")
include("unet.jl")

export UNet

end

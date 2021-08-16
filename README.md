# SegmentationModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://maxfreu.github.io/SegmentationModels.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maxfreu.github.io/SegmentationModels.jl/dev)
[![Build Status](https://github.com/maxfreu/SegmentationModels.jl/workflows/CI/badge.svg)](https://github.com/maxfreu/SegmentationModels.jl/actions)

The aim of this package is to create accomplish similar functionality to https://github.com/qubvel/segmentation_models.pytorch. Pull requests are very welcome.


## Supported Architectures
- UNet

## Supported Backbones
- VGG
- ResNet

A subset of the classifiers in [Metalhead.jl](https://github.com/FluxML/Metalhead.jl) is supported. 
Pre-training is supported as far as its supported there.


## Usage

```julia
using CUDA
using Flux
using SegmentationModels

data = rand(Float32, 256, 256, 8, 1) |> gpu
unet = UNet(8,1; init_channels=16, stages=4) |> gpu  # returns unet with simple double-conv backbone as a placeholder

# or
unet = UNet(ResNet50(;pretrain=true); num_classes=1337) |> gpu

unet(data)
```

## ToDo
- Add other segmentation architectures
- Add option to change the number of input channels even for 
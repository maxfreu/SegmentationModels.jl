# SegmentationModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://maxfreu.github.io/SegmentationModels.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maxfreu.github.io/SegmentationModels.jl/dev)
[![Build Status](https://github.com/maxfreu/SegmentationModels.jl/workflows/CI/badge.svg)](https://github.com/maxfreu/SegmentationModels.jl/actions)

This currently is a stub; there only is a simple, but flexible UNet implementation.

The intention is to create a package with similar functionality to https://github.com/qubvel/segmentation_models.pytorch. Pull requests are very welcome.


## Supported Architectures
- UNet

## Supported Backbones
- So far, only a custom one for the UNet
- No pretrained weights

## Usage

So far, you can only use a UNet. You can customize the number of stages (number of pooling operations), and the number of initial convolution channels. The number of channels gets doubled every stage. Every down and up stage involves two convolutions with BatchNorm.

```julia
using Flux
using SegmentationModels

data = rand(Float32, 256, 256, 8, 1) |> gpu
unet = UNet(8,1; init_channels=16, stages=4) |> gpu

unet(data)
```
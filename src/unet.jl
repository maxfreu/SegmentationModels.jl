function double_conv(in_channels, out_channels; activation=relu, bn_momentum=0.1f0)
    Chain(Conv((3,3), in_channels=>out_channels,  activation; pad=1),
          BatchNorm(out_channels; momentum=bn_momentum),
          Conv((3,3), out_channels=>out_channels, activation; pad=1),
          BatchNorm(out_channels; momentum=bn_momentum)
         )
end

struct UNetUpBlock
    upsampling_op
    conv_op
end

Flux.@functor UNetUpBlock

function UNetUpBlock(in_ch_up, in_ch_concat, out_ch; activation=relu, bn_momentum=0.1f0)
    up_op = Upsample(:bilinear; scale=2)
    conv_op = double_conv(in_ch_up + in_ch_concat, out_ch; activation=activation, bn_momentum=bn_momentum)
    UNetUpBlock(up_op, conv_op)
end

function Base.show(io::IO, b::UNetUpBlock)
    println(io, "UpBlock")
    println(io, b.upsampling_op)
    println(io, b.conv_op)
end

function (b::UNetUpBlock)(up_input, concat_input; dims=3)
    up = cat(b.upsampling_op(up_input), concat_input; dims=dims)
    return b.conv_op(up)
end

"""
    UNet(in_channels::Integer=3, num_classes::Integer=1; init_channels::Integer=16, stages::Integer=4, final_activation=sigmoid)

Instantiate a [UNet](https://arxiv.org/pdf/1505.04597.pdf) with a given number of input channels and output classes.
The inital number of convolution channels can be set and also the number of pooling operations (stages).
The final activation of the segmentation head can also be supplied.
The model applies two convolutions with relu activation and BatchNorm at each encoder or decoder stage.
It doubles the number of convolution channels at each down-stage.
Upsampling is done bilinear.
"""
struct UNet
    encoder
    decoder
    segmentation_head
end

Flux.@functor UNet

function UNet(in_channels::Integer=3, num_classes::Integer=1;
              init_channels::Integer=16,
              stages::Integer=4,
              final_activation=sigmoid)

    down_stage1 = double_conv(in_channels, init_channels)
    down_stages = ntuple(i -> Chain(MaxPool((2,2)),
                                    double_conv(init_channels*2^(i-1), init_channels*2^i)),
                        stages)
    
    encoder = Chain(down_stage1, down_stages...)
    decoder = ntuple(i -> UNetUpBlock(init_channels*2^(stages-i+1),
                                      init_channels*2^(stages-i),
                                      init_channels*2^(stages-i)),
                    stages)

    segmentation_head = Conv((1,1), init_channels=>num_classes, final_activation)

    UNet(encoder, decoder, segmentation_head)
end

function decode(ops::Tuple, ft::Tuple)
    up = first(ops)(ft[end], ft[end-1])
    decode(Base.tail(ops), (ft[1:end-2]..., up))
end

decode(::Tuple{}, ft::NTuple{1, T}) where T = first(ft)

function (u::UNet)(input)
    encoder_features = Flux.activations(u.encoder, input)
    up = decode(u.decoder, encoder_features)
    return u.segmentation_head(up)
end

function Base.show(io::IO, u::UNet)
    println(io, "UNet:")
    println(io, "\n")
    println(io, "Encoder:")
    for l in u.encoder
        println(io, l)
    end
    println(io, "\n")
    println(io, "Decoder:")
    for l in u.decoder
        println(io, l)
    end
    println(io, "Segmentation head:")
    println(io, u.segmentation_head)
end

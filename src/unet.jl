function double_conv(in_channels, out_channels; activation=relu, bn_momentum=0.1f0)
    Chain(Conv((3,3), in_channels=>out_channels,  activation; pad=1),
          BatchNorm(out_channels; momentum=bn_momentum),
          Conv((3,3), out_channels=>out_channels, activation; pad=1),
          BatchNorm(out_channels; momentum=bn_momentum)
         )
end

struct UNetUpBlock{U,C}
    upsampling_op::U
    conv_op::C
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
    UNet(m::Classifier; num_classes=1, decoder_channels=(16,32,64,128,256,512,1024), final_activation=sigmoid)

There are two options to instantiate a [UNet](https://arxiv.org/pdf/1505.04597.pdf):

1. with a given number of input channels and output classes
2. with a classifier from Metalhead, either ResNet or VGG

In the first case you get a UNet with a simple backbone using double convs, where you can specify the number of input channels.
In the second case, you can only specify the number of classes.
"""
struct UNet{E,D,S}
    encoder::E
    decoder::D
    segmentation_head::S
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

function UNet(m::Classifier; num_classes=1, decoder_channels=(16,32,64,128,256,512,1024), final_activation=sigmoid)
    enc = encoder(m)
    enc_channels = encoder_channels(enc)
    decoder_channels = decoder_channels[1:length(enc_channels)-1]
    decoder_channels = (decoder_channels..., last(enc_channels))
    
    decoder = ntuple(i -> UNetUpBlock(decoder_channels[i+1], enc_channels[i], decoder_channels[i]), length(enc_channels)-1)[end:-1:1]
    
    segmentation_head = Conv((1,1), decoder_channels[1]=>num_classes, final_activation)
    UNet(enc, decoder, segmentation_head)
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
    print(io, "\n")
    println(io, "Encoder:")
    Flux._big_show(io, u.encoder)
    
    println(io, "\n")
    println(io, "Decoder:")
    for l in u.decoder
        println(io, l)
    end
    println(io, "Segmentation head:")
    println(io, u.segmentation_head)
end

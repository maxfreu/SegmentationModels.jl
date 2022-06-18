
# trim off unused network parts
get_encoding_network_part(m::Union{ResNet, ResNeXt, VGG}) = m.layers[1][1:end-1]
get_encoding_network_part(m::MobileNetv1) = m.layers[1][1:end-2]
get_encoding_network_part(m::MobileNetv2) = m.layers[1][1:end-2][1]
get_encoding_network_part(m::MobileNetv3) = m.layers[1][1:end-2][1]

function get_encoding_network_part_with_fixed_input_channels(m, input_channels)
    layers = get_encoding_network_part(m).layers

    if input_channels != default_input_size(m)[3]
        new_conv = change_convlayer_input(layers[1], input_channels)
        layers[1] = new_conv
    end

    if prepend_identity(m)
        return Chain(Chain(identity), layers...)
    else
        return Chain(layers...)
    end
end

# if the first conv directly performs downsampling, we link the input directly to to the last decoder stage
# for that we have to prepend the identity layer
prepend_identity(::Classifier) = true
prepend_identity(::VGG) = false

default_input_size(::Classifier) = (256,256,3,1)

function outputsizes(fs::Tuple, x)
    res = Flux.outputsize(first(fs), x)
    return (res, outputsizes(Base.tail(fs), res)...)
end

outputsizes(::Tuple{}, x) = ()
outputsizes(c::Chain, x) = outputsizes(c.layers, x)

function encoder(m, input_channels)
    input_size = default_input_size(m)
    input_size = (input_size[1:2]..., input_channels, input_size[4])
    encoder = get_encoding_network_part_with_fixed_input_channels(m, input_channels)
    os = outputsizes(encoder, input_size)  # snoop intermediate output sizes
    os = map(x->x[1], os)  # take only the width
    os = (os..., 0)  # the below loop checks for differences in size, so we have to set an endpoint to also catch the last layers

    layers = []
    last_size = os[1]
    last = 1
    for i in 1:length(os)
        current_size = os[i]
        if current_size != last_size
            push!(layers, encoder[last:i-1])
            last = i
            last_size = current_size
        end
    end
    return Chain(layers...)
end

function encoder_channels(m, input_channels=3)
    os = outputsizes(m, (512,512,input_channels,1))
    return getindex.(os, 3)
end

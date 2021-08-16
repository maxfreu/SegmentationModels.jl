
function outputsizes(fs::Tuple, x)
    res = Flux.outputsize(first(fs), x)
    return (res, outputsizes(Base.tail(fs), res)...)
end
outputsizes(::Tuple{}, x) = ()
outputsizes(c::Chain, x) = outputsizes(c.layers, x)

# trim off unused network parts
get_encoding_network_part(m::ResNet) = Chain(Chain(identity), m.layers[1][1:end-1]...)
get_encoding_network_part(m::VGG) = m.layers[1][1:end-1]

function encoder(m)
    encoder = get_encoding_network_part(m)
    os = outputsizes(encoder, (512,512,3,1))  # snoop intermediate output sizes
    os = map(x->x[1], os)  # take only the width
    os = (os..., 0)  # the below loop checks for differences in size, so we have to set an endpoint to also catch the last layers

    layers = []
    last_size = os[1]
    last = 1
    for i in 1:length(os)
        current_size = os[i]
        if current_size != last_size
            push!(layers, deepcopy(encoder[last:i-1]))  # actually I don't know if the deepcopy is needed, but just to be sure...
            last = i
            last_size = current_size
        end
    end
    return Chain(layers...)
end

function encoder_channels(m, input_channels=3)
    os = outputsizes(m, (512, 512, input_channels, 1))
    return map(x->x[3], os)
end

function change_convlayer_input(cl::Conv, new_in_channels::Integer)
    w,h,in,out = size(cl.weight)
    new_conv = Conv((w,h), new_in_channels => out, cl.σ; stride=cl.stride, pad=cl.pad, dilation=cl.dilation, groups=cl.groups, bias=cl.bias)
    if new_in_channels > in
        new_conv.weight[:,:,1:in,:] .= cl.weight
    elseif new_in_channels < in
        @views new_conv.weight .= cl.weight[:,:,1:new_in_channels,:]
    end
    return new_conv
end


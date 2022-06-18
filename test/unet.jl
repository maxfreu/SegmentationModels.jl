@testset "UNet" begin
    x = rand(Float32, 256,256,3,1)
    os = (256,256,1,1)
    @testset "ResNet" begin
        u = UNet(ResNet())
        @test size(u(x)) == os
    end
    @testset "ResNeXt" begin
        u = UNet(ResNeXt())
        @test size(u(x)) == os
    end
    @testset "VGG" begin
        u = UNet(VGG())
        @test size(u(x)) == os
    end
    @testset "MobileNetv1" begin
        u = UNet(MobileNetv1())
        @test size(u(x)) == os
    end
    @testset "MobileNetv2" begin
        u = UNet(MobileNetv1())
        @test size(u(x)) == os
    end
    @testset "MobileNetv3" begin
        u = UNet(MobileNetv1())
        @test size(u(x)) == os
    end
end
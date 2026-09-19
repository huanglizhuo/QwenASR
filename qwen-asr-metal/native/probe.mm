// Metal 4 runtime compiler smoke test. This is not an ASR implementation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>

int main() {
    @autoreleasepool {
        if (@available(macOS 26.4, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
                fprintf(stderr, "Metal 4 Apple Silicon device required\n"); return 2;
            }
            NSString *source = @R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void tensor_probe(device half *a [[buffer(0)]],
                         device half *b [[buffer(1)]],
                         device float *c [[buffer(2)]]) {
    auto A = tensor(a, dextents<int,2>{32,32}, array<int,2>{1,32});
    auto B = tensor(b, dextents<int,2>{32,32}, array<int,2>{1,32});
    auto C = tensor(c, dextents<int,2>{32,32}, array<int,2>{1,32});
    constexpr auto desc = matmul2d_descriptor(32,32);
    matmul2d<desc, execution_simdgroup> op;
    auto result = op.get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
    op.run(A, B, result);
    result.store(C);
}
)metal";
            NSError *error = nil;
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) {
                fprintf(stderr, "TensorOps runtime compilation failed: %s\n", error.description.UTF8String); return 3;
            }
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tensor_probe"] error:&error];
            if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 4; }
            id<MTLBuffer> a = [device newBufferWithLength:2048 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b = [device newBufferWithLength:2048 options:MTLResourceStorageModeShared];
            id<MTLBuffer> c = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
            if (!a || !b || !c) return 5;
            auto ap = (__fp16 *)a.contents; auto bp = (__fp16 *)b.contents;
            for (int i=0;i<1024;++i) { ap[i] = (__fp16)((i % 11 - 5) / 8.0f); bp[i] = (__fp16)((i % 7 - 3) / 4.0f); }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a offset:0 atIndex:0];
            [encoder setBuffer:b offset:0 atIndex:1];
            [encoder setBuffer:c offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "GPU execution failed: %s\n", command.error.description.UTF8String); return 6;
            }
            float max_error = 0;
            for (int m=0;m<32;++m) for (int n=0;n<32;++n) {
                float expected = 0;
                for (int k=0;k<32;++k) expected += (float)ap[m*32+k] * (float)bp[k*32+n];
                float actual = ((float *)c.contents)[m*32+n];
                if (!std::isfinite(actual)) return 7;
                max_error = fmaxf(max_error, fabsf(actual - expected));
            }
            NSDictionary *result = @{@"device": device.name, @"metal4": @YES,
                                    @"tensorops_fp16_smoke_passed": @(max_error <= 1e-4f),
                                    @"max_abs_error": @(max_error),
                                    @"asr_implemented": @NO, @"align_implemented": @NO};
            NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];
            puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
            return max_error <= 1e-4f ? 0 : 8;
        } else {
            fprintf(stderr, "macOS 26.4+ required\n"); return 2;
        }
    }
}

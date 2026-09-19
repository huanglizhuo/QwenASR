// Synthetic matrix-path benchmark; never reports end-to-end ASR speedup.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

struct Shape { uint32_t m, n, k; };
struct Work { const char *name; Shape shape; };
static double millis() {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(),v.end()); return v[v.size()/2];
}
static id<MTLBuffer> buffer(id<MTLDevice> d, size_t bytes) {
    id<MTLBuffer> b = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("GPU allocation failed");
    return b;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        try {
            if (argc != 3) throw std::runtime_error("usage: metal-microbench <quantized.metal> <output.json>");
            if (@available(macOS 26.4, *)) {} else { throw std::runtime_error("macOS 26.4+ required"); }
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) throw std::runtime_error("Metal 4 device required");
            NSError *error = nil;
            NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
            if (!source) throw std::runtime_error(error.description.UTF8String);
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) throw std::runtime_error(error.description.UTF8String);
            NSMutableDictionary *pipelines = [NSMutableDictionary dictionary];
            for (NSString *name in @[@"gemv_w8a16",@"gemv_w4a16",@"tensor_w8a16",@"row_scale"]) {
                id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
                if (!p) throw std::runtime_error(error.description.UTF8String);
                pipelines[name] = p;
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            const Work work[] = {
                {"decoder_qkv", {1,4096,1024}}, {"decoder_gate_up", {1,6144,1024}},
                {"decoder_down", {1,1024,3072}}, {"asr_lm_head", {1,151936,1024}},
                {"batched_qkv", {8,4096,1024}}, {"asr_prefill_qkv", {128,4096,1024}},
                {"asr_encoder_fc1", {128,3584,896}}, {"align_encoder_fc1", {128,4096,1024}},
                {"align_timestamp_head", {40,5000,1024}}, {"encoder_conv_projection", {13,896,7680}},
            };
            NSMutableArray *results = [NSMutableArray array];
            for (const auto &test : work) {
                @autoreleasepool {
                    Shape s = test.shape;
                    id<MTLBuffer> x = buffer(device, size_t(s.m)*s.k*2);
                    id<MTLBuffer> w8 = buffer(device, size_t(s.n)*s.k);
                    id<MTLBuffer> w4 = buffer(device, size_t(s.n)*s.k/2);
                    id<MTLBuffer> scale = buffer(device, size_t(s.n)*4);
                    id<MTLBuffer> out = buffer(device, size_t(s.m)*s.n*4);
                    auto xp = (__fp16 *)x.contents; auto wp = (int8_t *)w8.contents;
                    auto w4p = (uint8_t *)w4.contents; auto sp = (float *)scale.contents;
                    for (size_t i=0;i<size_t(s.m)*s.k;++i) xp[i] = (__fp16)((int((i*17+13)%37)-18)/32.0f);
                    // Both paths use the SAME exactly representable logical
                    // weights in [-8,7], so format/runtime effects are isolated.
                    // This synthetic distribution is not a model-quality test.
                    for (size_t i=0;i<size_t(s.n)*s.k;++i) wp[i] = int8_t(int((i*11+i/31)%16)-8);
                    for (size_t i=0;i<size_t(s.n)*s.k;i+=2) w4p[i/2] = (wp[i]&15) | ((wp[i+1]&15)<<4);
                    for (uint32_t i=0;i<s.n;++i) sp[i] = float(i%4+1)/32;
                    for (NSString *name in @[@"gemv_w8a16",@"gemv_w4a16",@"tensor_w8a16"]) {
                        bool tensor = [name isEqualToString:@"tensor_w8a16"];
                        bool four = [name isEqualToString:@"gemv_w4a16"];
                        // GEMV is a low-batch candidate, not a deliberately
                        // poor full-prefill baseline.
                        if (!tensor && s.m > 8) continue;
                        std::vector<double> wall, gpu;
                        for (int iteration=-2;iteration<11;++iteration) {
                            double start = millis();
                            id<MTLCommandBuffer> command = [queue commandBuffer];
                            command.label = [NSString stringWithFormat:@"%s/%@", test.name,name];
                            id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
                            [e setComputePipelineState:pipelines[name]];
                            [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:four?w4:w8 offset:0 atIndex:1];
                            [e setBuffer:out offset:0 atIndex:2]; [e setBuffer:scale offset:0 atIndex:3];
                            [e setBytes:&s length:sizeof(s) atIndex:4];
                            [e dispatchThreadgroups:MTLSizeMake((s.n+(tensor?31:7))/(tensor?32:8), (s.m+(tensor?31:0))/(tensor?32:1), 1)
                               threadsPerThreadgroup:MTLSizeMake(tensor?32:256,1,1)];
                            if (tensor) {
                                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                                [e setComputePipelineState:pipelines[@"row_scale"]];
                                [e setBuffer:out offset:0 atIndex:0]; [e setBuffer:scale offset:0 atIndex:1];
                                [e setBytes:&s length:sizeof(s) atIndex:2];
                                [e dispatchThreads:MTLSizeMake(size_t(s.m)*s.n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                            }
                            [e endEncoding]; [command commit]; [command waitUntilCompleted];
                            if (command.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(command.error.description.UTF8String);
                            if (iteration >= 0) {
                                wall.push_back(millis()-start);
                                gpu.push_back((command.GPUEndTime-command.GPUStartTime)*1000);
                            }
                        }
                        // Check boundary rows/columns as well as spread samples
                        // to catch tile, transpose, packing and output-stride bugs.
                        double max_error=0;
                        for (uint32_t m : {0u,s.m/2,s.m-1}) for (uint32_t j=0;j<35;++j) {
                            uint32_t n = j==0 ? 0 : j==1 ? s.n-1 : (j*7919)%s.n;
                            double expected=0;
                            for (uint32_t k=0;k<s.k;++k) expected += float(xp[m*s.k+k]) * int(wp[n*s.k+k]);
                            expected *= sp[n];
                            float actual = ((float *)out.contents)[m*s.n+n];
                            if (!std::isfinite(actual)) throw std::runtime_error("nonfinite matrix output");
                            max_error=std::max(max_error,std::abs(expected-actual));
                        }
                        if (max_error>1e-3) throw std::runtime_error("matrix numerical check failed");
                        [results addObject:@{@"name":@(test.name),@"kernel":name,@"m":@(s.m),@"n":@(s.n),@"k":@(s.k),
                                             @"median_host_ms":@(median(wall)),@"median_gpu_ms":@(median(gpu)),
                                             @"weight_bytes":@(size_t(s.n)*s.k/(four?2:1)),@"max_abs_error":@(max_error),
                                             @"measured_runs":@11,@"warmups":@2}];
                        fprintf(stderr,"%s / %s: %.4f ms GPU, %.4f ms host\n",test.name,name.UTF8String,median(gpu),median(wall));
                    }
                }
            }
            NSDictionary *report = @{@"schema":@1,@"device":device.name,@"results":results,
                                     @"kind":@"synthetic_matrix_microbenchmark",@"end_to_end_goal_verified":@NO,
                                     @"note":@"Hot resident weights; synthetic identical integer matrices; not a quality test or ASR speedup claim."};
            NSData *data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
            if (!data || ![data writeToFile:@(argv[2]) options:NSDataWritingAtomic error:&error]) throw std::runtime_error(error.description.UTF8String);
            return 0;
        } catch (const std::exception &e) {
            fprintf(stderr,"microbenchmark failed: %s\n",e.what()); return 2;
        }
    }
}

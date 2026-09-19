#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Shape { uint m, n, k; };

// W8A16, row-scaled weights, FP32 reduction. One SIMD group per output.
kernel void gemv_w8a16(device half *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
                      device float *out [[buffer(2)]], device float *scales [[buffer(3)]],
                      constant Shape &s [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    uint row = group.x * 8 + tid / 32;
    uint lane = tid % 32;
    if (row >= s.n) return;
    float sum = 0;
    for (uint k = lane; k < s.k; k += 32)
        sum += float(x[group.y * s.k + k]) * float(w[row * s.k + k]);
    sum = simd_sum(sum);
    if (lane == 0) out[group.y * s.n + row] = sum * scales[row];
}

// W4A16 uses signed two's-complement nibbles, even K element in low bits.
kernel void gemv_w4a16(device half *x [[buffer(0)]], device uchar *w [[buffer(1)]],
                      device float *out [[buffer(2)]], device float *scales [[buffer(3)]],
                      constant Shape &s [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    uint row = group.x * 8 + tid / 32;
    uint lane = tid % 32;
    if (row >= s.n) return;
    float sum = 0;
    for (uint k = lane; k < s.k; k += 32) {
        uint index = row * s.k + k;
        uint nibble = (w[index / 2] >> ((index % 2) * 4)) & 15;
        int value = int(nibble ^ 8) - 8;
        sum += float(x[group.y * s.k + k]) * float(value);
    }
    sum = simd_sum(sum);
    if (lane == 0) out[group.y * s.n + row] = sum * scales[row];
}

// Dynamic slices preserve bounds for M=1 and N=5000 edge tiles.
// The separate scale epilogue is included in the measured command buffer.
kernel void tensor_w8a16(device half *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
                        device float *out [[buffer(2)]], device float *scales [[buffer(3)]],
                        constant Shape &s [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]]) {
    auto X = tensor(x, dextents<int,2>{int(s.k),int(s.m)}, array<int,2>{1,int(s.k)});
    auto W = tensor(w, dextents<int,2>{int(s.k),int(s.n)}, array<int,2>{1,int(s.k)});
    auto O = tensor(out, dextents<int,2>{int(s.n),int(s.m)}, array<int,2>{1,int(s.n)});
    auto a = X.slice(0, group.y * 32);
    auto b = W.slice(0, group.x * 32);
    auto c = O.slice(group.x * 32, group.y * 32);
    constexpr auto desc = matmul2d_descriptor(32,32,int(dynamic_extent),false,true,false);
    matmul2d<desc, execution_simdgroup> op;
    auto result = op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,result);
    result.store(c);
}

kernel void row_scale(device float *out [[buffer(0)]], device float *scales [[buffer(1)]],
                      constant Shape &s [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < s.m * s.n) out[index] *= scales[index % s.n];
}

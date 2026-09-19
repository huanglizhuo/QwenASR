#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct BufferLease;
struct BufferPool;
struct Tensor {
    id<MTLBuffer> buffer = nil;
    size_t offset = 0;
    std::vector<size_t> shape;
    std::string dtype;
    std::string scale;
    std::shared_ptr<BufferLease> lease;
    size_t elements() const;
};

// Owns one immutable, mmap-backed package and its shared Metal buffer.
class ModelPackage {
    void *mapping_ = nullptr;
    size_t bytes_ = 0;
    id<MTLBuffer> weights_ = nil;
    std::map<std::string, Tensor> tensors_;
public:
    NSDictionary *index = nil;
    ModelPackage(id<MTLDevice> device, const std::string &directory);
    ~ModelPackage();
    ModelPackage(const ModelPackage&) = delete;
    ModelPackage &operator=(const ModelPackage&) = delete;
    const Tensor &weight(const std::string &name) const;
    bool has(const std::string &name) const { return tensors_.count(name) != 0; }
};

class MetalRuntime {
    id<MTLLibrary> library_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLCommandBuffer> command_ = nil;
    id<MTLComputeCommandEncoder> encoder_ = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipelines_;
    Tensor dft_cos_, dft_sin_, hann_, mel_filters_;
    std::shared_ptr<BufferPool> pool_;
    id<MTLCounterSampleBuffer> profile_samples_ = nil;
    std::vector<std::string> profile_names_;
    std::string profile_path_;
    std::vector<id<MTLCommandBuffer>> chain_;
    unsigned dispatch_count_ = 0;
    unsigned chunk_threshold_ = 64;
    void bind(const Tensor &tensor, unsigned slot);
    void dispatch(const std::string &name, MTLSize groups, unsigned threads);
    void create_encoder();
public:
    std::pair<Tensor,Tensor> quantize_decode_input(const Tensor &input);
    id<MTLDevice> device = nil;
    MetalRuntime(const std::string &kernel_directory);
    ~MetalRuntime();
    MetalRuntime(const MetalRuntime&) = delete;
    MetalRuntime &operator=(const MetalRuntime&) = delete;
    Tensor allocate(std::vector<size_t> shape, const std::string &dtype="f32", bool host_write=false);
    Tensor upload(const float *values, size_t rows, size_t columns);
    void begin();
    void submit(); // commit the current chunk; GPU starts while the host encodes on
    double finish(); // waits for completion, reports GPU ms; never enqueue-only
    void enable_profiling(const std::string &path); // development probe only; production never calls this
    Tensor linear(const Tensor &input, const ModelPackage &model, const std::string &weight);
    // emit_half: elementwise producers store half output directly, feeding
    // linear() without the cast_half pass (identical f16 bits, one less trip).
    Tensor linear_half(const Tensor &input, const ModelPackage &model, const std::string &weight);
    std::vector<Tensor> decoder_qkv(const Tensor &input, const ModelPackage &model, const std::string &layer_prefix);
    Tensor decoder_gate(const Tensor &input, const ModelPackage &model, const std::string &layer_prefix);
    Tensor norm(const Tensor &input, const Tensor &weight, const Tensor *bias, float epsilon, bool emit_half=false);
    Tensor activation(const Tensor &input, unsigned operation); // 0=GELU, 1=SiLU
    Tensor add(const Tensor &a, const Tensor &b);
    Tensor bias_gelu(const Tensor &input, const Tensor *bias, bool gelu, bool emit_half=false);
    Tensor mel_chunk(const Tensor &mel, size_t start, size_t width);
    Tensor im2col(const Tensor &input, size_t height, size_t width, bool emit_half=false);
    Tensor flatten_conv(const Tensor &input, size_t height, size_t width, bool emit_half=false);
    Tensor position(const Tensor &input, bool emit_half=false);
    Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, unsigned heads, unsigned window, bool emit_half=false);
    void split_qkv_bias(const Tensor &fused, const Tensor &bias, Tensor &q, Tensor &k, Tensor &v);
    Tensor embedding(const ModelPackage &model, const std::vector<int32_t> &tokens);
    Tensor head_norm_rope(const Tensor &x, const Tensor &weight, unsigned heads, unsigned start, float epsilon, float theta);
    Tensor causal_attention(const Tensor &q, const Tensor &k_cache, const Tensor &v_cache, unsigned start, bool emit_half=false);
    // Fused single-token decode helpers; each reproduces its unfused sequence
    // bit-exactly (see decoder.metal).
    std::pair<Tensor,Tensor> norm_quant(const Tensor &x, const Tensor &weight, float epsilon);
    Tensor qkv_w8a8(const Tensor &qx, const Tensor &x_scale, const ModelPackage &model, const std::string &layer_prefix);
    Tensor gate_up_w8a8(const Tensor &qx, const Tensor &x_scale, const ModelPackage &model, const std::string &layer_prefix);
    Tensor rope_cache(const Tensor &qkv, const ModelPackage &model, const std::string &layer_prefix,
                      const Tensor &k_cache, const Tensor &v_cache, unsigned slot, float epsilon, float theta);
    std::pair<Tensor,Tensor> causal_attention_quant(const Tensor &q, const Tensor &k_cache, const Tensor &v_cache, unsigned start);
    Tensor gemv_w8a8(const Tensor &qx, const Tensor &x_scale, const ModelPackage &model, const std::string &weight);
    // Batched speculative-verify projection: m>1 rows through the hybrid
    // .decode INT8 weights with the tensor_w8a16 path (f16 activations).
    Tensor linear_decode(const Tensor &x, const ModelPackage &model, const std::string &weight);
    Tensor gemv_residual(const Tensor &qx, const Tensor &x_scale, const ModelPackage &model,
                         const std::string &weight, const Tensor &residual);
    Tensor swiglu(const Tensor &gate, const Tensor &up, bool emit_half=false);
    Tensor argmax(const Tensor &logits);
    Tensor gather_rows(const Tensor &x, const std::vector<int32_t> &rows);
    Tensor mel_spectrogram(const Tensor &pcm);
    void copy(const Tensor &source, const Tensor &destination);
};

Tensor encode_mel(MetalRuntime &runtime, const ModelPackage &model, const Tensor &mel);

// hybrid* packages run the single-token decode quantization paths.
static inline bool hybrid_decode(NSDictionary *index){
    NSString *precision=index[@"precision"];
    return [precision isEqual:@"hybrid"] || [precision isEqual:@"hybrid4"];
}

// One request owns its KV state. reset() logically discards prior positions;
// every position is overwritten on GPU before any subsequent attention reads it.
struct DecoderCache {
    std::vector<Tensor> keys, values;
    size_t length = 0, capacity = 0;
    DecoderCache(MetalRuntime &runtime, size_t capacity);
    void reset() { length=0; }
};
Tensor decoder_forward(MetalRuntime &runtime, const ModelPackage &model, DecoderCache &cache, const Tensor &input);
Tensor decoder_logits(MetalRuntime &runtime, const ModelPackage &model, const Tensor &hidden);
// Speculative verify batch: m rows (draft tokens) through the .decode INT8
// weights, KV written at cache.length; returns hidden rows. cache.length is
// advanced by the CALLER by the accepted count.
Tensor decoder_forward_decode_batch(MetalRuntime &runtime, const ModelPackage &model, DecoderCache &cache, const Tensor &input);
Tensor decoder_logits_decode_batch(MetalRuntime &runtime, const ModelPackage &model, const Tensor &hidden);

#include "runtime.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <CommonCrypto/CommonDigest.h>

static void require(bool ok, const std::string &why) { if(!ok) throw std::runtime_error(why); }
// A lease is independent of Metal's command-buffer retains. Once the final
// Tensor view dies, later GPU commands may reuse that buffer: the encoder is
// explicitly serial, every resource is tracked, and command buffers complete
// before the next submission. Metal supplies the required dependency barriers.
// CPU-written uploads NEVER enter this pool, since CPU writes are not ordered
// by a GPU barrier and could overwrite input for an earlier queued dispatch.
struct BufferPool {
    std::map<size_t,std::vector<id<MTLBuffer>>> free;
    size_t cached_bytes=0;
};
struct BufferLease {
    id<MTLBuffer> buffer;
    std::shared_ptr<BufferPool> pool;
    size_t bytes;
    BufferLease(id<MTLBuffer> b,std::shared_ptr<BufferPool> p,size_t n):buffer(b),pool(std::move(p)),bytes(n){}
    ~BufferLease(){
        if(pool->cached_bytes+bytes<=256u*1024*1024){
            try{pool->free[bytes].push_back(buffer);pool->cached_bytes+=bytes;}catch(...){/* release instead of caching */}
        }
    }
};
static size_t item_size(const std::string &dtype) {
    if(dtype=="f32" || dtype=="i32") return 4;
    if(dtype=="f16") return 2;
    if(dtype=="i8") return 1;
    if(dtype=="i4") return 0; // packed; use storage_bytes()
    throw std::runtime_error("unsupported dtype: "+dtype);
}
static size_t storage_bytes(size_t count,const std::string &dtype) {
    if(dtype=="i4") return (count+1)/2;
    size_t unit=item_size(dtype);
    require(count<=SIZE_MAX/unit,"tensor byte overflow");
    return count*unit;
}
size_t Tensor::elements() const {
    size_t count=1;
    for(size_t n:shape) {
        require(n>0 && count<=SIZE_MAX/n,"invalid/overflowing tensor shape"); count*=n;
    }
    return count;
}

ModelPackage::ModelPackage(id<MTLDevice> device,const std::string &directory) {
    @autoreleasepool {
        NSError *error=nil;
        NSData *data=[NSData dataWithContentsOfFile:@((directory+"/index.json").c_str())];
        require(data!=nil,"cannot read package index");
        index=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        require([index isKindOfClass:NSDictionary.class],"invalid package JSON");
        require([index[@"format"] isEqual:@"qwen-asr-metal-v1"],"unsupported model format");
        require([index[@"precision"] isEqual:@"fp16"] || [index[@"precision"] isEqual:@"int8"] || hybrid_decode(index),"unsupported package precision");
        int fd=open((directory+"/weights.bin").c_str(),O_RDONLY);
        require(fd>=0,"cannot open weights.bin");
        struct stat st{};
        bool valid=fstat(fd,&st)==0 && st.st_size>0 && size_t(st.st_size)==[index[@"weights_bytes"] unsignedLongLongValue]
                   && size_t(st.st_size)%getpagesize()==0 && size_t(st.st_size)<=device.maxBufferLength;
        if(!valid) {close(fd);throw std::runtime_error("invalid weights file size/alignment");}
        bytes_=size_t(st.st_size);
        mapping_=mmap(nullptr,bytes_,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);
        close(fd);
        if(mapping_==MAP_FAILED) {mapping_=nullptr;throw std::runtime_error("weights mmap failed");}
        try {
            require([index[@"weights_sha256"] isKindOfClass:NSString.class],"missing weights checksum");
            // Package verification belongs to loading, never to timed inference.
            CC_SHA256_CTX hash;CC_SHA256_Init(&hash);
            for(size_t offset=0;offset<bytes_;){
                size_t n=std::min(size_t(1<<20),bytes_-offset);
                CC_SHA256_Update(&hash,(const char *)mapping_+offset,CC_LONG(n));offset+=n;
            }
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&hash);
            char encoded[65];for(size_t i=0;i<32;++i)snprintf(encoded+2*i,3,"%02x",digest[i]);
            require([index[@"weights_sha256"] isEqualToString:@(encoded)],"weights checksum mismatch");
            weights_=[device newBufferWithBytesNoCopy:mapping_ length:bytes_ options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked deallocator:nil];
            require(weights_!=nil,"Metal no-copy weight buffer failed");
            NSDictionary *entries=index[@"tensors"];
            require([entries isKindOfClass:NSDictionary.class],"missing tensor index");
            for(NSString *name in entries) {
                NSDictionary *entry=entries[name];
                require([entry[@"dtype"] isKindOfClass:NSString.class] && [entry[@"shape"] isKindOfClass:NSArray.class],"invalid tensor metadata");
                Tensor t; t.buffer=weights_;t.offset=[entry[@"offset"] unsignedLongLongValue]; t.dtype=[entry[@"dtype"] UTF8String];
                for(NSNumber *dim in entry[@"shape"]) t.shape.push_back(dim.unsignedLongLongValue);
                require(!t.shape.empty(),"empty tensor shape");
                size_t count=t.elements();
                size_t n=storage_bytes(count,t.dtype);
                require(t.offset%256==0 && t.offset<=bytes_ && n<=bytes_-t.offset && n==[entry[@"bytes"] unsignedLongLongValue],"tensor outside package bounds");
                if(entry[@"scale"])t.scale=[entry[@"scale"] UTF8String];
                tensors_.emplace(name.UTF8String,t);
            }
            for(const auto &[name,t]:tensors_) if(t.dtype=="i8" || t.dtype=="i4") {
                const auto &scale=weight(t.scale);
                size_t per_row=t.dtype=="i8"?1u:(t.shape[1]+63)/64;
                require(t.shape.size()==2 && scale.dtype=="f32" && scale.elements()==t.shape[0]*per_row,"invalid quantized scales");
            }
        } catch(...) {
            tensors_.clear(); weights_=nil;munmap(mapping_,bytes_);mapping_=nullptr;throw;
        }
    }
}
ModelPackage::~ModelPackage(){tensors_.clear();weights_=nil;if(mapping_)munmap(mapping_,bytes_);}
const Tensor &ModelPackage::weight(const std::string &name) const {
    auto it=tensors_.find(name);require(it!=tensors_.end(),"missing model tensor: "+name);return it->second;
}

MetalRuntime::MetalRuntime(const std::string &directory) {
    pool_=std::make_shared<BufferPool>();
    chunk_threshold_=64;
    if(@available(macOS 26.4,*)){}else throw std::runtime_error("macOS 26.4+ required");
    device=MTLCreateSystemDefaultDevice();
    require(device && [device supportsFamily:MTLGPUFamilyMetal4],"Metal 4 GPU required");
    NSError *error=nil;
    NSMutableString *source=[NSMutableString string];
    for(const char *name:{"quantized.metal","ops.metal","decoder.metal","frontend.metal"}) {
        NSString *part=[NSString stringWithContentsOfFile:@((directory+"/"+name).c_str()) encoding:NSUTF8StringEncoding error:&error];
        require(part!=nil,"cannot read shader file");[source appendString:part];[source appendString:@"\n"];
    }
    MTLCompileOptions *options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;
    library_=[device newLibraryWithSource:source options:options error:&error];
    require(library_!=nil,error ? error.description.UTF8String:"shader compilation failed");
    for(const char *name:{"cast_half","tensor_f16","tensor_w8a16","row_scale","norm_f32","activate_f32","add_f32",
                         "bias_gelu_f32","mel_chunk_f32","im2col_f32","flatten_conv_f32","position_f32","window_attention_f32","copy_f32",
                         "gemv_f16","gemv_w8a16","embedding_f32","head_norm_rope_f32","causal_attention_f32","swiglu_f32","argmax_f32","gather_rows_f32",
                         "frontend_tables","dft_power_f32","mel_log_f32","mel_max_f32","mel_normalize_f32","quantize_activation_i8","gemv_w8a8","qkv_w8a8","gate_up_w8a8","argmax_blocks_f32","argmax_finish_f32",
                         "norm_quant_i8","rope_cache_f32","causal_attention_quant_i8","gemv_w8a8_residual","mel_max_blocks_f32","mel_max_finish_f32",
                         "norm_f32_h","bias_gelu_f32_h","position_f32_h","window_attention_f32_h","im2col_f32_h","flatten_conv_f32_h","causal_attention_f32_h","swiglu_f32_h","split_qkv_bias_f32","split_qkv_f32","split_swiglu_h","gemv_w4a8_residual","gate_up_w4a8"}) {
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:[library_ newFunctionWithName:@(name)] error:&error];
        require(p!=nil,error ? error.description.UTF8String:"pipeline creation failed");pipelines_[name]=p;
    }
    queue_=[device newCommandQueue];require(queue_!=nil,"command queue creation failed");
    dft_cos_=allocate({201,400});dft_sin_=allocate({201,400});hann_=allocate({400});mel_filters_=allocate({128,201});
    begin();bind(dft_cos_,0);bind(dft_sin_,1);bind(hann_,2);bind(mel_filters_,3);
    dispatch("frontend_tables",MTLSizeMake((201*400+255)/256,1,1),256);finish();
}
// Abandon uncommitted work on exception; a model may already have been unmapped
// by stack unwinding. Never submit such commands from a destructor.
MetalRuntime::~MetalRuntime(){if(encoder_)[encoder_ endEncoding];encoder_=nil;command_=nil;}
Tensor MetalRuntime::allocate(std::vector<size_t> shape,const std::string &dtype,bool host_write){
    Tensor t;t.shape=std::move(shape);t.dtype=dtype;
    size_t count=t.elements();require(count<=device.maxBufferLength/item_size(dtype),"tensor allocation exceeds device limit");
    size_t bytes=(count*item_size(dtype)+255)&~size_t(255);
    if(!host_write){
        auto it=pool_->free.find(bytes);
        if(it!=pool_->free.end() && !it->second.empty()){
            t.buffer=it->second.back();it->second.pop_back();pool_->cached_bytes-=bytes;
        }
    }
    if(!t.buffer)t.buffer=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];
    require(t.buffer!=nil,"GPU tensor allocation failed");
    if(!host_write)t.lease=std::make_shared<BufferLease>(t.buffer,pool_,bytes);
    return t;
}
Tensor MetalRuntime::upload(const float *values,size_t rows,size_t columns){
    auto t=allocate({rows,columns},"f32",true);memcpy(t.buffer.contents,values,t.elements()*4);return t;
}
void MetalRuntime::begin(){
    require(!command_,"command already active");command_=[queue_ commandBuffer];
    chain_.clear();dispatch_count_=0;
    profile_names_.clear();
    if(profile_path_!=""){
        // A fresh sample buffer per command buffer: reusing slots across
        // command buffers made resolveCounterRange return stale samples.
        id<MTLCounterSet> timestamps=nil;
        for(id<MTLCounterSet> set in device.counterSets)if([set.name isEqual:MTLCommonCounterSetTimestamp])timestamps=set;
        require(timestamps!=nil,"GPU timestamp counter set unavailable");
        auto descriptor=[MTLCounterSampleBufferDescriptor new];descriptor.counterSet=timestamps;
        descriptor.storageMode=MTLStorageModeShared;descriptor.sampleCount=4096;
        NSError *error=nil;profile_samples_=[device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        require(profile_samples_!=nil,error?error.description.UTF8String:"GPU sample buffer failed");
    }
    create_encoder();
}
void MetalRuntime::submit(){
    // Pipeline the CPU encoder against GPU execution: finish the current
    // command buffer so the GPU can start while the host encodes the next
    // chunk. The serial queue plus tracked resources preserve the exact
    // RAW/WAR/WAW ordering the single-buffer serial encoder had.
    require(command_!=nil,"no active GPU commands");
    [encoder_ endEncoding];encoder_=nil;
    [command_ commit];chain_.push_back(command_);command_=nil;
    command_=[queue_ commandBuffer];dispatch_count_=0;
    create_encoder();
}
void MetalRuntime::create_encoder(){
    if(profile_samples_){
        require(profile_names_.size()*2+2<=profile_samples_.sampleCount,"GPU profiling sample capacity exceeded");
        auto descriptor=[MTLComputePassDescriptor computePassDescriptor];descriptor.dispatchType=MTLDispatchTypeSerial;
        descriptor.sampleBufferAttachments[0].sampleBuffer=profile_samples_;
        descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex=profile_names_.size()*2;
        descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex=profile_names_.size()*2+1;
        encoder_=[command_ computeCommandEncoderWithDescriptor:descriptor];
    }else encoder_=[command_ computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    require(encoder_!=nil,"cannot begin GPU commands");
}
double MetalRuntime::finish(){
    require(command_!=nil,"no active GPU commands");[encoder_ endEncoding];encoder_=nil;
    [command_ commit];chain_.push_back(command_);
    for(id<MTLCommandBuffer> buffer:chain_)[buffer waitUntilCompleted];
    bool ok=true;std::string message="GPU execution failed";double total=0;
    for(id<MTLCommandBuffer> buffer:chain_){
        ok=ok && buffer.status==MTLCommandBufferStatusCompleted;
        if(buffer.error)message=buffer.error.description.UTF8String;
        total+=(buffer.GPUEndTime-buffer.GPUStartTime)*1000;
    }
    command_=nil;chain_.clear();
    require(ok,message);
    if(profile_samples_ && !profile_names_.empty()){
        NSData *data=[profile_samples_ resolveCounterRange:NSMakeRange(0,profile_names_.size()*2)];
        require(data && data.length==profile_names_.size()*2*sizeof(MTLCounterResultTimestamp),"cannot resolve GPU counters");
        const auto *values=(const MTLCounterResultTimestamp *)data.bytes;
        NSMutableDictionary *summary=[NSMutableDictionary dictionary];
        for(size_t i=0;i<profile_names_.size();++i){
            uint64_t a=values[2*i].timestamp,b=values[2*i+1].timestamp;
            require(a!=MTLCounterErrorValue && b!=MTLCounterErrorValue && b>=a,"invalid GPU timestamp sample");
            NSString *name=@(profile_names_[i].c_str());NSDictionary *prior=summary[name];
            summary[name]=@{@"count":@([prior[@"count"] unsignedLongLongValue]+1),@"ticks":@([prior[@"ticks"] unsignedLongLongValue]+b-a)};
        }
        NSData *json=[NSJSONSerialization dataWithJSONObject:summary options:NSJSONWritingSortedKeys error:nil];
        std::ofstream file(profile_path_,std::ios::app);file.write((const char *)json.bytes,json.length);file<<'\n';
        require(bool(file),"cannot write profile counters");
    }
    return total;
}
void MetalRuntime::enable_profiling(const std::string &path){
    require(!command_,"cannot enable profiling during a command");
    require([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary],"GPU stage counters unsupported");
    id<MTLCounterSet> timestamps=nil;
    for(id<MTLCounterSet> set in device.counterSets)if([set.name isEqual:MTLCommonCounterSetTimestamp])timestamps=set;
    require(timestamps!=nil,"GPU timestamp counter set unavailable");
    auto descriptor=[MTLCounterSampleBufferDescriptor new];descriptor.counterSet=timestamps;descriptor.storageMode=MTLStorageModeShared;descriptor.sampleCount=4096;
    NSError *error=nil;profile_samples_=[device newCounterSampleBufferWithDescriptor:descriptor error:&error];
    require(profile_samples_!=nil,error?error.description.UTF8String:"GPU sample buffer failed");profile_path_=path;
}
void MetalRuntime::bind(const Tensor &t,unsigned slot){
    require(encoder_!=nil && t.buffer!=nil,"missing encoder or tensor");[encoder_ setBuffer:t.buffer offset:t.offset atIndex:slot];
}
void MetalRuntime::dispatch(const std::string &name,MTLSize groups,unsigned threads){
    auto it=pipelines_.find(name);require(it!=pipelines_.end(),"missing pipeline");
    [encoder_ setComputePipelineState:it->second];
    [encoder_ dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    if(profile_samples_){
        // Apple devices expose stage counters, so the development probe places
        // each kernel in a pass. This changes scheduling and is never used to
        // certify performance; it identifies the dominant arithmetic stages.
        [encoder_ endEncoding];
        profile_names_.push_back(name);
        create_encoder();
    } else if(++dispatch_count_>=chunk_threshold_){
        // Auto-chunk long command buffers so GPU execution overlaps host
        // encoding (see submit()). Disabled while profiling.
        submit();
    }
    // Serial dispatch + tracked resources provides RAW/WAR/WAW ordering. A
    // global explicit barrier here redundantly drains all buffers after every
    // small kernel. Concurrent/untracked encoders MUST NOT use this policy.
}
Tensor MetalRuntime::linear_half(const Tensor &half_in,const ModelPackage &model,const std::string &name){
    // half_in already carries the exact bits cast_half would produce.
    const auto &w=model.weight(name);
    require(half_in.dtype=="f16" && half_in.shape.size()==2 && w.shape.size()>=2,"invalid half linear input/weight");
    size_t k=w.elements()/w.shape[0],n=w.shape[0],m=half_in.shape[0];
    require(k==half_in.shape[1] && m<=UINT32_MAX && n<=UINT32_MAX && k<=UINT32_MAX && half_in.elements()<=UINT32_MAX,"linear shape mismatch/overflow");
    require(w.dtype=="f16","half linear requires f16 weights");
    auto out=allocate({m,n});
    struct {uint32_t m,n,k;} shape{uint32_t(m),uint32_t(n),uint32_t(k)};
    bind(half_in,0);bind(w,1);bind(out,2);[encoder_ setBytes:&shape length:sizeof(shape) atIndex:4];
    if(m==1){
        dispatch("gemv_f16",MTLSizeMake((n+7)/8,1,1),256);
        return out;
    }
    dispatch("tensor_f16",MTLSizeMake((n+31)/32,(m+31)/32,1),32);
    return out;
}
Tensor MetalRuntime::linear(const Tensor &x,const ModelPackage &model,const std::string &name){
    if(x.dtype=="f16")return linear_half(x,model,name);
    bool hybrid=x.shape.size()==2 && x.shape[0]==1 && hybrid_decode(model.index) &&
                (name.rfind("thinker.model.layers.",0)==0 || name=="thinker.lm_head.weight");
    const auto &w=model.weight(hybrid?name+".decode":name);
    require(x.dtype=="f32" && x.shape.size()==2 && w.shape.size()>=2,"invalid linear input/weight");
    size_t k=w.elements()/w.shape[0],n=w.shape[0],m=x.shape[0];
    require(k==x.shape[1] && m<=UINT32_MAX && n<=UINT32_MAX && k<=UINT32_MAX && x.elements()<=UINT32_MAX,"linear shape mismatch/overflow");
    if(hybrid){
        require(w.dtype=="i8" && k%128==0,"invalid hybrid decode matrix");
        auto [q,scale]=quantize_decode_input(x);auto out=allocate({1,n});
        struct{uint32_t m,n,k;}shape{1,uint32_t(n),uint32_t(k)};
        bind(q,0);bind(w,1);bind(out,2);bind(model.weight(w.scale),3);bind(scale,5);[encoder_ setBytes:&shape length:sizeof(shape) atIndex:4];
        dispatch("gemv_w8a8",MTLSizeMake((n+7)/8,1,1),256);return out;
    }
    auto half=allocate(x.shape,"f16"),out=allocate({m,n});uint32_t count=uint32_t(x.elements());
    bind(x,0);bind(half,1);[encoder_ setBytes:&count length:4 atIndex:2];
    dispatch("cast_half",MTLSizeMake((count+255)/256,1,1),256);
    struct {uint32_t m,n,k;} shape{uint32_t(m),uint32_t(n),uint32_t(k)};
    bind(half,0);bind(w,1);bind(out,2);[encoder_ setBytes:&shape length:sizeof(shape) atIndex:4];
    require(w.dtype=="i8" || w.dtype=="f16","unsupported linear weight precision");
    if(w.dtype=="i8")bind(model.weight(w.scale),3);
    if(m==1){
        dispatch(w.dtype=="i8"?"gemv_w8a16":"gemv_f16",MTLSizeMake((n+7)/8,1,1),256);
        return out;
    }
    dispatch(w.dtype=="i8"?"tensor_w8a16":"tensor_f16",MTLSizeMake((n+31)/32,(m+31)/32,1),32);
    if(w.dtype=="i8") {
        bind(out,0);bind(model.weight(w.scale),1);[encoder_ setBytes:&shape length:sizeof(shape) atIndex:2];
        dispatch("row_scale",MTLSizeMake((m*n+255)/256,1,1),256);
    }
    return out;
}
std::pair<Tensor,Tensor> MetalRuntime::quantize_decode_input(const Tensor &x){
    require(x.dtype=="f32" && x.shape.size()==2 && x.shape[0]==1 && x.shape[1]<=UINT32_MAX,"invalid decode activation");
    auto q=allocate(x.shape,"i8"),scale=allocate({1});uint32_t width=uint32_t(x.shape[1]);
    bind(x,0);bind(q,1);bind(scale,2);[encoder_ setBytes:&width length:4 atIndex:3];
    dispatch("quantize_activation_i8",MTLSizeMake(1,1,1),256);return {q,scale};
}
std::vector<Tensor> MetalRuntime::decoder_qkv(const Tensor &x,const ModelPackage &model,const std::string &p){
    if(x.shape.size()!=2 || x.shape[0]!=1 || !hybrid_decode(model.index)){
        if(x.shape[0]!=1 && model.has(p+"self_attn.fused_qkv_prefill.weight")){
            // One fused GEMM over concatenated q/k/v; identical per-column
            // reductions. Split into contiguous q/k/v (views would have the
            // wrong row stride for m>1).
            auto fused=linear(x,model,p+"self_attn.fused_qkv_prefill.weight");
            size_t rows=fused.shape[0];
            auto q=allocate({rows,2048}),k=allocate({rows,1024}),v=allocate({rows,1024});
            uint32_t sp[2]={uint32_t(rows),1024u};
            bind(fused,0);bind(q,1);bind(k,2);bind(v,3);[encoder_ setBytes:sp length:sizeof(sp) atIndex:4];
            dispatch("split_qkv_f32",MTLSizeMake((rows*2048+255)/256,1,1),256);
            return {q,k,v};
        }
        return {linear(x,model,p+"self_attn.q_proj.weight"),linear(x,model,p+"self_attn.k_proj.weight"),linear(x,model,p+"self_attn.v_proj.weight")};
    }
    require(x.shape[1]==1024,"invalid fused QKV input");auto [qx,xs]=quantize_decode_input(x);
    auto out=allocate({1,4096});bind(qx,0);bind(xs,7);bind(out,8);
    unsigned slot=1;
    for(const char *projection:{"q_proj","k_proj","v_proj"}){
        const auto &w=model.weight(p+"self_attn."+projection+".weight.decode");
        require(w.dtype=="i8" && w.shape==std::vector<size_t>{slot==1?2048u:1024u,1024},"invalid fused QKV weight");
        bind(w,slot);bind(model.weight(w.scale),slot+3);++slot;
    }
    dispatch("qkv_w8a8",MTLSizeMake(4096/8,1,1),256);
    auto q=out,k=out,v=out;q.shape[1]=2048;k.shape[1]=v.shape[1]=1024;k.offset=2048*4;v.offset=3072*4;
    return {q,k,v};
}
Tensor MetalRuntime::decoder_gate(const Tensor &x,const ModelPackage &model,const std::string &p){
    if(x.shape.size()!=2 || x.shape[0]!=1 || !hybrid_decode(model.index)){
        if(x.shape[0]!=1 && model.has(p+"mlp.fused_gate_up_prefill.weight")){
            auto fused=linear(x,model,p+"mlp.fused_gate_up_prefill.weight");
            size_t rows=fused.shape[0];
            auto out=allocate({rows,3072},"f16");
            uint32_t sp[2]={uint32_t(rows),3072u};
            bind(fused,0);bind(out,1);[encoder_ setBytes:sp length:sizeof(sp) atIndex:2];
            dispatch("split_swiglu_h",MTLSizeMake((rows*3072+255)/256,1,1),256);
            return out;
        }
        return swiglu(linear(x,model,p+"mlp.gate_proj.weight"),linear(x,model,p+"mlp.up_proj.weight"),true);
    }
    require(x.shape[1]==1024,"invalid fused FFN input");auto [qx,xs]=quantize_decode_input(x);auto out=allocate({1,3072});
    const auto &g=model.weight(p+"mlp.gate_proj.weight.decode"),&u=model.weight(p+"mlp.up_proj.weight.decode");
    require(g.dtype=="i8" && u.dtype=="i8" && g.shape==u.shape && g.shape==std::vector<size_t>{3072,1024},"invalid fused FFN weights");
    bind(qx,0);bind(g,1);bind(u,2);bind(model.weight(g.scale),3);bind(model.weight(u.scale),4);bind(xs,5);bind(out,6);
    dispatch("gate_up_w8a8",MTLSizeMake(3072/8,1,1),256);return out;
}
Tensor MetalRuntime::norm(const Tensor &x,const Tensor &weight,const Tensor *bias,float epsilon,bool emit_half){
    require(x.dtype=="f32" && x.shape.size()==2 && weight.dtype=="f32" && weight.elements()==x.shape[1],"invalid norm shape");
    if(bias)require(bias->dtype=="f32" && bias->elements()==x.shape[1],"invalid norm bias");
    bool half=emit_half;auto out=allocate(x.shape,half?"f16":"f32");
    struct {uint32_t width;float epsilon;uint32_t layer;} p{uint32_t(x.shape[1]),epsilon,bias?1u:0u};
    bind(x,0);bind(weight,1);bind(bias?*bias:weight,2);bind(out,3);[encoder_ setBytes:&p length:sizeof(p) atIndex:4];
    dispatch(half?"norm_f32_h":"norm_f32",MTLSizeMake(x.shape[0],1,1),32);return out;
}
Tensor MetalRuntime::activation(const Tensor &x,unsigned operation){
    require(x.dtype=="f32" && operation<=1 && x.elements()<=UINT32_MAX,"invalid activation");auto out=allocate(x.shape);
    struct {uint32_t count,operation;} p{uint32_t(x.elements()),operation};
    bind(x,0);bind(out,1);[encoder_ setBytes:&p length:sizeof(p) atIndex:2];
    dispatch("activate_f32",MTLSizeMake((p.count+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::add(const Tensor &a,const Tensor &b){
    require(a.dtype=="f32" && b.dtype=="f32" && a.shape==b.shape && a.elements()<=UINT32_MAX,"invalid residual shapes");
    auto out=allocate(a.shape);uint32_t count=uint32_t(a.elements());
    bind(a,0);bind(b,1);bind(out,2);[encoder_ setBytes:&count length:4 atIndex:3];
    dispatch("add_f32",MTLSizeMake((count+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::bias_gelu(const Tensor &x,const Tensor *bias,bool gelu,bool emit_half){
    require(x.dtype=="f32" && x.shape.size()==2 && x.elements()<=UINT32_MAX,"invalid bias input");
    if(bias)require(bias->dtype=="f32" && bias->elements()==x.shape[1],"invalid bias shape");
    bool half=emit_half;auto out=allocate(x.shape,half?"f16":"f32");
    struct{uint32_t count,width,has_bias,gelu;}p{uint32_t(x.elements()),uint32_t(x.shape[1]),bias?1u:0u,gelu?1u:0u};
    bind(x,0);bind(bias?*bias:x,1);bind(out,2);[encoder_ setBytes:&p length:sizeof(p) atIndex:3];
    dispatch(half?"bias_gelu_f32_h":"bias_gelu_f32",MTLSizeMake((p.count+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::mel_chunk(const Tensor &mel,size_t start,size_t width){
    require(mel.dtype=="f32" && mel.shape.size()==2 && mel.shape[0]==128 && width>0 && start+width<=mel.shape[1],"invalid mel chunk");
    auto out=allocate({128*width,1});
    struct{uint32_t frames,start,width;}p{uint32_t(mel.shape[1]),uint32_t(start),uint32_t(width)};
    bind(mel,0);bind(out,1);[encoder_ setBytes:&p length:sizeof(p) atIndex:2];
    dispatch("mel_chunk_f32",MTLSizeMake((out.elements()+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::im2col(const Tensor &x,size_t height,size_t width,bool emit_half){
    require(x.dtype=="f32" && x.shape.size()==2 && x.shape[0]==height*width,"invalid im2col shape");
    size_t oh=(height+1)/2,ow=(width+1)/2;
    bool half=emit_half;auto out=allocate({oh*ow,x.shape[1]*9},half?"f16":"f32");
    struct{uint32_t h,w,c,oh,ow;}p{uint32_t(height),uint32_t(width),uint32_t(x.shape[1]),uint32_t(oh),uint32_t(ow)};
    bind(x,0);bind(out,1);[encoder_ setBytes:&p length:sizeof(p) atIndex:2];
    dispatch(half?"im2col_f32_h":"im2col_f32",MTLSizeMake((out.elements()+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::flatten_conv(const Tensor &x,size_t height,size_t width,bool emit_half){
    require(x.dtype=="f32" && x.shape.size()==2 && x.shape[0]==height*width,"invalid conv flatten shape");
    bool half=emit_half;auto out=allocate({width,height*x.shape[1]},half?"f16":"f32");
    struct{uint32_t h,w,c,oh,ow;}p{uint32_t(height),uint32_t(width),uint32_t(x.shape[1]),0,0};
    bind(x,0);bind(out,1);[encoder_ setBytes:&p length:sizeof(p) atIndex:2];
    dispatch(half?"flatten_conv_f32_h":"flatten_conv_f32",MTLSizeMake((out.elements()+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::position(const Tensor &x,bool emit_half){
    require(x.dtype=="f32" && x.shape.size()==2 && x.shape[1]>2 && x.shape[1]%2==0,"invalid position shape");
    bool half=emit_half;auto out=allocate(x.shape,half?"f16":"f32");uint32_t p[2]={uint32_t(x.shape[0]),uint32_t(x.shape[1])};
    bind(x,0);bind(out,1);[encoder_ setBytes:p length:sizeof(p) atIndex:2];
    dispatch(half?"position_f32_h":"position_f32",MTLSizeMake((out.elements()+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::attention(const Tensor &q,const Tensor &k,const Tensor &v,unsigned heads,unsigned window,bool emit_half){
    require(q.dtype=="f32" && k.dtype=="f32" && v.dtype=="f32" && q.shape==k.shape && q.shape==v.shape && q.shape.size()==2 && heads>0 && window>0,"invalid attention tensors");
    unsigned dim=unsigned(q.shape[1]/heads);
    require(q.shape[1]%heads==0 && (dim==64 || dim==128),"unsupported attention head dimension");
    bool half=emit_half;auto out=allocate(q.shape,half?"f16":"f32");struct{uint32_t seq,heads,dim,window;}p{uint32_t(q.shape[0]),heads,dim,window};
    bind(q,0);bind(k,1);bind(v,2);bind(out,3);[encoder_ setBytes:&p length:sizeof(p) atIndex:4];
    dispatch(half?"window_attention_f32_h":"window_attention_f32",MTLSizeMake(q.shape[0],heads,1),32);return out;
}
void MetalRuntime::split_qkv_bias(const Tensor &fused,const Tensor &bias,Tensor &q,Tensor &k,Tensor &v){
    require(fused.dtype=="f32" && fused.shape.size()==2 && fused.shape[1]%3==0 && bias.dtype=="f32" && bias.elements()==fused.shape[1]
            && q.dtype=="f32" && k.dtype=="f32" && v.dtype=="f32" && q.shape==std::vector<size_t>({fused.shape[0],fused.shape[1]/3})
            && k.shape==q.shape && v.shape==q.shape,"invalid fused qkv split");
    struct{uint32_t rows,src_width,dst_width,has_bias;}p{uint32_t(fused.shape[0]),uint32_t(fused.shape[1]),uint32_t(fused.shape[1]/3),1u};
    bind(fused,0);bind(bias,1);bind(q,2);bind(k,3);bind(v,4);[encoder_ setBytes:&p length:sizeof(p) atIndex:5];
    dispatch("split_qkv_bias_f32",MTLSizeMake((fused.shape[0]*p.dst_width+255)/256,1,1),256);
}
void MetalRuntime::copy(const Tensor &source,const Tensor &destination){
    require(source.dtype=="f32" && destination.dtype=="f32" && source.shape==destination.shape && source.elements()<=UINT32_MAX,"invalid copy");
    uint32_t count=uint32_t(source.elements());bind(source,0);bind(destination,1);[encoder_ setBytes:&count length:4 atIndex:2];
    dispatch("copy_f32",MTLSizeMake((count+255)/256,1,1),256);
}

Tensor MetalRuntime::embedding(const ModelPackage &model,const std::vector<int32_t> &tokens){
    const auto &w=model.weight("thinker.model.embed_tokens.weight");
    require(w.shape.size()==2 && (w.dtype=="i8" || w.dtype=="f16") && !tokens.empty(),"invalid embedding request");
    for(int32_t t:tokens)require(t>=0 && size_t(t)<w.shape[0],"embedding token outside vocabulary");
    auto ids=allocate({tokens.size()},"i32",true);memcpy(ids.buffer.contents,tokens.data(),tokens.size()*4);
    auto out=allocate({tokens.size(),w.shape[1]});
    require(out.elements()<=UINT32_MAX,"embedding size overflow");
    struct{uint32_t rows,width,quantized;}p{uint32_t(tokens.size()),uint32_t(w.shape[1]),w.dtype=="i8"?1u:0u};
    bind(w,0);bind(w.dtype=="i8"?model.weight(w.scale):out,1);bind(ids,2);bind(out,3);
    [encoder_ setBytes:&p length:sizeof(p) atIndex:4];
    dispatch("embedding_f32",MTLSizeMake((out.elements()+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::head_norm_rope(const Tensor &x,const Tensor &w,unsigned heads,unsigned start,float epsilon,float theta){
    require(x.dtype=="f32" && x.shape.size()==2 && (heads==8 || heads==16) && x.shape[1]==heads*128 && w.dtype=="f32" && w.elements()==128 && start+x.shape[0]<=65536,"invalid head norm/RoPE");
    auto out=allocate(x.shape);struct{uint32_t heads,start;float epsilon,theta;}p{heads,start,epsilon,theta};
    bind(x,0);bind(w,1);bind(out,2);[encoder_ setBytes:&p length:sizeof(p) atIndex:3];
    dispatch("head_norm_rope_f32",MTLSizeMake(x.shape[0]*heads,1,1),32);return out;
}
Tensor MetalRuntime::causal_attention(const Tensor &q,const Tensor &k,const Tensor &v,unsigned start,bool emit_half){
    require(q.dtype=="f32" && k.dtype=="f32" && v.dtype=="f32" && q.shape.size()==2 && q.shape[1]==2048 && k.shape==v.shape && k.shape.size()==2 && k.shape[1]==1024 && start+q.shape[0]<=k.shape[0],"invalid causal attention");
    bool half=emit_half;auto out=allocate(q.shape,half?"f16":"f32");uint32_t p[2]={uint32_t(q.shape[0]),start};
    bind(q,0);bind(k,1);bind(v,2);bind(out,3);[encoder_ setBytes:p length:sizeof(p) atIndex:4];
    dispatch(half?"causal_attention_f32_h":"causal_attention_f32",MTLSizeMake(q.shape[0],16,1),32);return out;
}
std::pair<Tensor,Tensor> MetalRuntime::norm_quant(const Tensor &x,const Tensor &weight,float epsilon){
    require(x.dtype=="f32" && x.shape==std::vector<size_t>{1,1024} && weight.dtype=="f32" && weight.elements()==1024 && epsilon>0,"invalid fused norm/quant");
    auto q=allocate({1,1024},"i8"),scale=allocate({1});
    bind(x,0);bind(weight,1);bind(q,2);bind(scale,3);[encoder_ setBytes:&epsilon length:4 atIndex:4];
    dispatch("norm_quant_i8",MTLSizeMake(1,1,1),32);return {q,scale};
}
Tensor MetalRuntime::qkv_w8a8(const Tensor &qx,const Tensor &x_scale,const ModelPackage &model,const std::string &p){
    require(qx.dtype=="i8" && qx.shape==std::vector<size_t>{1,1024} && x_scale.shape==std::vector<size_t>{1},"invalid fused qkv input");
    auto out=allocate({1,4096});bind(qx,0);bind(x_scale,7);bind(out,8);
    unsigned slot=1;
    for(const char *projection:{"q_proj","k_proj","v_proj"}){
        const auto &w=model.weight(p+"self_attn."+projection+".weight.decode");
        require(w.dtype=="i8" && w.shape==std::vector<size_t>{slot==1?2048u:1024u,1024},"invalid fused QKV weight");
        bind(w,slot);bind(model.weight(w.scale),slot+3);++slot;
    }
    dispatch("qkv_w8a8",MTLSizeMake(4096/8,1,1),256);return out;
}
Tensor MetalRuntime::linear_decode(const Tensor &x,const ModelPackage &model,const std::string &name){
    const auto &w=model.weight(name+".decode");
    require(x.dtype=="f32" && x.shape.size()==2 && x.shape[0]>1 && w.dtype=="i8" && w.shape.size()==2
            && w.shape[1]==x.shape[1],"invalid batched decode projection");
    size_t m=x.shape[0],n=w.shape[0],k=w.shape[1];
    auto half=allocate(x.shape,"f16"),out=allocate({m,n});uint32_t count=uint32_t(x.elements());
    bind(x,0);bind(half,1);[encoder_ setBytes:&count length:4 atIndex:2];
    dispatch("cast_half",MTLSizeMake((count+255)/256,1,1),256);
    struct {uint32_t m,n,k;} shape{uint32_t(m),uint32_t(n),uint32_t(k)};
    bind(half,0);bind(w,1);bind(out,2);[encoder_ setBytes:&shape length:sizeof(shape) atIndex:4];
    bind(model.weight(w.scale),3);
    dispatch("tensor_w8a16",MTLSizeMake((n+31)/32,(m+31)/32,1),32);
    bind(out,0);bind(model.weight(w.scale),1);[encoder_ setBytes:&shape length:sizeof(shape) atIndex:2];
    dispatch("row_scale",MTLSizeMake((m*n+255)/256,1,1),256);
    return out;
}
Tensor MetalRuntime::gate_up_w8a8(const Tensor &qx,const Tensor &x_scale,const ModelPackage &model,const std::string &p){
    require(qx.dtype=="i8" && qx.shape==std::vector<size_t>{1,1024} && x_scale.shape==std::vector<size_t>{1},"invalid fused gate/up input");
    auto out=allocate({1,3072});
    const auto &g=model.weight(p+"mlp.gate_proj.weight.decode"),&u=model.weight(p+"mlp.up_proj.weight.decode");
    require((g.dtype=="i8" || g.dtype=="i4") && g.dtype==u.dtype && g.shape==u.shape && g.shape==std::vector<size_t>{3072,1024},"invalid fused FFN weights");
    bind(qx,0);bind(g,1);bind(u,2);bind(model.weight(g.scale),3);bind(model.weight(u.scale),4);bind(x_scale,5);bind(out,6);
    dispatch(g.dtype=="i4"?"gate_up_w4a8":"gate_up_w8a8",MTLSizeMake(3072/8,1,1),256);return out;
}
Tensor MetalRuntime::rope_cache(const Tensor &qkv,const ModelPackage &model,const std::string &p,
                                const Tensor &k_cache,const Tensor &v_cache,unsigned slot,float epsilon,float theta){
    require(qkv.dtype=="f32" && qkv.shape==std::vector<size_t>{1,4096} && k_cache.dtype=="f32" && v_cache.dtype=="f32"
            && k_cache.shape==std::vector<size_t>{v_cache.shape[0],1024} && slot<k_cache.shape[0],"invalid rope/cache request");
    auto q=allocate({1,2048});
    struct{uint32_t slot;float epsilon,theta;}args{slot,epsilon,theta};
    bind(qkv,0);bind(q,1);bind(k_cache,2);bind(v_cache,3);
    bind(model.weight(p+"self_attn.q_norm.weight"),4);bind(model.weight(p+"self_attn.k_norm.weight"),5);
    [encoder_ setBytes:&args length:sizeof(args) atIndex:6];
    dispatch("rope_cache_f32",MTLSizeMake(32,1,1),32);return q;
}
std::pair<Tensor,Tensor> MetalRuntime::causal_attention_quant(const Tensor &q,const Tensor &k,const Tensor &v,unsigned start){
    require(q.dtype=="f32" && q.shape==std::vector<size_t>{1,2048} && k.dtype=="f32" && v.dtype=="f32"
            && k.shape==v.shape && k.shape.size()==2 && k.shape[1]==1024 && start+1<=k.shape[0],"invalid fused attention");
    auto q8=allocate({1,2048},"i8"),scale=allocate({1});
    bind(q,0);bind(k,1);bind(v,2);bind(q8,3);bind(scale,4);[encoder_ setBytes:&start length:4 atIndex:5];
    dispatch("causal_attention_quant_i8",MTLSizeMake(1,1,1),512);return {q8,scale};
}
Tensor MetalRuntime::gemv_w8a8(const Tensor &qx,const Tensor &x_scale,const ModelPackage &model,const std::string &name){
    const auto &w=model.weight(name+".decode");
    require(qx.dtype=="i8" && qx.shape[0]==1 && x_scale.shape==std::vector<size_t>{1} && w.dtype=="i8" && w.shape.size()==2
            && w.shape[1]==qx.shape[1] && w.shape[1]%128==0,"invalid gemv request");
    auto out=allocate({1,w.shape[0]});
    struct{uint32_t m,n,k;}shape{1,uint32_t(w.shape[0]),uint32_t(w.shape[1])};
    bind(qx,0);bind(w,1);bind(out,2);bind(model.weight(w.scale),3);
    [encoder_ setBytes:&shape length:sizeof(shape) atIndex:4];bind(x_scale,5);
    dispatch("gemv_w8a8",MTLSizeMake((w.shape[0]+7)/8,1,1),256);return out;
}
Tensor MetalRuntime::gemv_residual(const Tensor &qx,const Tensor &x_scale,const ModelPackage &model,
                                   const std::string &name,const Tensor &residual){
    const auto &w=model.weight(name+".decode");
    require(qx.dtype=="i8" && qx.shape[0]==1 && x_scale.shape==std::vector<size_t>{1} && (w.dtype=="i8" || (w.dtype=="i4" && w.shape[1]%8==0)) && w.shape.size()==2
            && w.shape[1]==qx.shape[1] && residual.dtype=="f32" && residual.shape==std::vector<size_t>{1,w.shape[0]},"invalid fused gemv/residual");
    auto out=allocate({1,w.shape[0]});
    struct{uint32_t m,n,k;}shape{1,uint32_t(w.shape[0]),uint32_t(w.shape[1])};
    bind(qx,0);bind(w,1);bind(residual,2);bind(out,3);bind(model.weight(w.scale),4);
    [encoder_ setBytes:&shape length:sizeof(shape) atIndex:5];bind(x_scale,6);
    dispatch(w.dtype=="i4"?"gemv_w4a8_residual":"gemv_w8a8_residual",MTLSizeMake((w.shape[0]+7)/8,1,1),256);return out;
}
Tensor MetalRuntime::swiglu(const Tensor &gate,const Tensor &up,bool emit_half){
    require(gate.dtype=="f32" && up.dtype=="f32" && gate.shape==up.shape && gate.elements()<=UINT32_MAX,"invalid SwiGLU");
    bool half=emit_half;auto out=allocate(gate.shape,half?"f16":"f32");uint32_t count=uint32_t(out.elements());
    bind(gate,0);bind(up,1);bind(out,2);[encoder_ setBytes:&count length:4 atIndex:3];
    dispatch(half?"swiglu_f32_h":"swiglu_f32",MTLSizeMake((count+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::argmax(const Tensor &logits){
    require(logits.dtype=="f32" && logits.shape.size()==2 && logits.elements()<=UINT32_MAX,"invalid logits");
    auto out=allocate({logits.shape[0]},"i32");uint32_t width=uint32_t(logits.shape[1]);
    if(width>4096){
        uint32_t blocks=(width+1023)/1024,p[2]={width,blocks};
        auto maxima=allocate({logits.shape[0],blocks}),indices=allocate({logits.shape[0],blocks},"i32");
        bind(logits,0);bind(maxima,1);bind(indices,2);[encoder_ setBytes:p length:sizeof(p) atIndex:3];
        dispatch("argmax_blocks_f32",MTLSizeMake(blocks,logits.shape[0],1),32);
        bind(maxima,0);bind(indices,1);bind(out,2);[encoder_ setBytes:&blocks length:4 atIndex:3];
        dispatch("argmax_finish_f32",MTLSizeMake(logits.shape[0],1,1),32);return out;
    }
    bind(logits,0);bind(out,1);[encoder_ setBytes:&width length:4 atIndex:2];
    dispatch("argmax_f32",MTLSizeMake(logits.shape[0],1,1),32);return out;
}
Tensor MetalRuntime::gather_rows(const Tensor &x,const std::vector<int32_t> &rows){
    require(x.dtype=="f32" && x.shape.size()==2 && !rows.empty(),"invalid row gather");
    for(int32_t r:rows)require(r>=0 && size_t(r)<x.shape[0],"gather row outside tensor");
    auto ids=allocate({rows.size()},"i32",true);memcpy(ids.buffer.contents,rows.data(),rows.size()*4);
    auto out=allocate({rows.size(),x.shape[1]});require(out.elements()<=UINT32_MAX,"gather size overflow");
    uint32_t p[2]={uint32_t(rows.size()),uint32_t(x.shape[1])};
    bind(x,0);bind(ids,1);bind(out,2);[encoder_ setBytes:p length:sizeof(p) atIndex:3];
    dispatch("gather_rows_f32",MTLSizeMake((out.elements()+255)/256,1,1),256);return out;
}
Tensor MetalRuntime::mel_spectrogram(const Tensor &pcm){
    require(pcm.dtype=="f32" && pcm.elements()>=160 && pcm.elements()<=480000,"frontend supports 10 ms to 30 s at 16 kHz");
    uint32_t frames=uint32_t(pcm.elements()/160),count=128*frames;
    auto power=allocate({frames,201}),mel=allocate({128,frames}),maximum=allocate({1});
    uint32_t p[2]={uint32_t(pcm.elements()),frames};
    bind(pcm,0);bind(dft_cos_,1);bind(dft_sin_,2);bind(hann_,3);bind(power,4);[encoder_ setBytes:p length:sizeof(p) atIndex:5];
    dispatch("dft_power_f32",MTLSizeMake(frames,201,1),32);
    bind(power,0);bind(mel_filters_,1);bind(mel,2);[encoder_ setBytes:&frames length:4 atIndex:3];
    dispatch("mel_log_f32",MTLSizeMake(frames,128,1),32);
    uint32_t blocks=(count+8191)/8192;
    auto partials=allocate({blocks});
    uint32_t bp[2]={count,blocks};
    bind(mel,0);bind(partials,1);[encoder_ setBytes:bp length:sizeof(bp) atIndex:2];
    dispatch("mel_max_blocks_f32",MTLSizeMake(blocks,1,1),256);
    bind(partials,0);bind(maximum,1);[encoder_ setBytes:&blocks length:4 atIndex:2];
    dispatch("mel_max_finish_f32",MTLSizeMake(1,1,1),32);
    bind(mel,0);bind(maximum,1);[encoder_ setBytes:&count length:4 atIndex:2];
    dispatch("mel_normalize_f32",MTLSizeMake((count+255)/256,1,1),256);return mel;
}

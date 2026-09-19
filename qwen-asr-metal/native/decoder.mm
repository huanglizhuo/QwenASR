#include "runtime.hpp"
#include <stdexcept>

DecoderCache::DecoderCache(MetalRuntime &rt,size_t size):capacity(size){
    if(size==0 || size>65536)throw std::runtime_error("invalid decoder capacity");
    for(unsigned i=0;i<28;++i){keys.push_back(rt.allocate({size,1024}));values.push_back(rt.allocate({size,1024}));}
}
Tensor decoder_forward(MetalRuntime &rt,const ModelPackage &model,DecoderCache &cache,const Tensor &input){
    NSDictionary *cfg=model.index[@"config"][@"thinker_config"][@"text_config"];
    if([cfg[@"hidden_size"] unsignedIntValue]!=1024 || [cfg[@"num_hidden_layers"] unsignedIntValue]!=28 ||
       [cfg[@"num_attention_heads"] unsignedIntValue]!=16 || [cfg[@"num_key_value_heads"] unsignedIntValue]!=8 ||
       [cfg[@"head_dim"] unsignedIntValue]!=128 || input.dtype!="f32" || input.shape.size()!=2 || input.shape[1]!=1024)
        throw std::runtime_error("unsupported decoder configuration/input");
    if(cache.length>cache.capacity || input.shape[0]>cache.capacity-cache.length)throw std::runtime_error("decoder KV capacity exceeded");
    float epsilon=[cfg[@"rms_norm_eps"] floatValue],theta=[cfg[@"rope_theta"] floatValue];
    if(epsilon<=0 || theta<=0)throw std::runtime_error("invalid decoder normalization/RoPE config");
    auto x=input;
    bool fused=input.shape[0]==1 && hybrid_decode(model.index);
    for(unsigned layer=0;layer<28;++layer){
        std::string p="thinker.model.layers."+std::to_string(layer)+".";
        if(fused){
            // Single-token path: nine kernels per layer instead of fifteen.
            // Each fused kernel reproduces its unfused sequence bit-exactly;
            // gemv_w8a8_residual blocks FMA contraction for the same rounding.
            auto [qx,xs]=rt.norm_quant(x,model.weight(p+"input_layernorm.weight"),epsilon);
            auto qkv=rt.qkv_w8a8(qx,xs,model,p);
            auto q=rt.rope_cache(qkv,model,p,cache.keys[layer],cache.values[layer],unsigned(cache.length),epsilon,theta);
            auto [aq,as]=rt.causal_attention_quant(q,cache.keys[layer],cache.values[layer],unsigned(cache.length));
            x=rt.gemv_residual(aq,as,model,p+"self_attn.o_proj.weight",x);
            auto [nq,ns]=rt.norm_quant(x,model.weight(p+"post_attention_layernorm.weight"),epsilon);
            auto [gq,gs]=rt.quantize_decode_input(rt.gate_up_w8a8(nq,ns,model,p));
            x=rt.gemv_residual(gq,gs,model,p+"mlp.down_proj.weight",x);
            continue;
        }
        bool rows=input.shape[0]>1; // single-token W8A8 needs f32 activations to quantize
        auto n=rt.norm(x,model.weight(p+"input_layernorm.weight"),nullptr,epsilon,rows);
        auto qkv=rt.decoder_qkv(n,model,p);
        auto q=rt.head_norm_rope(qkv[0],model.weight(p+"self_attn.q_norm.weight"),16,unsigned(cache.length),epsilon,theta);
        auto k=rt.head_norm_rope(qkv[1],model.weight(p+"self_attn.k_norm.weight"),8,unsigned(cache.length),epsilon,theta);
        auto v=qkv[2];
        auto kd=cache.keys[layer],vd=cache.values[layer];
        kd.offset+=cache.length*1024*4;vd.offset+=cache.length*1024*4;
        kd.shape=k.shape;vd.shape=v.shape;rt.copy(k,kd);rt.copy(v,vd);
        auto a=rt.causal_attention(q,cache.keys[layer],cache.values[layer],unsigned(cache.length),rows);
        x=rt.add(x,rt.linear(a,model,p+"self_attn.o_proj.weight"));
        n=rt.norm(x,model.weight(p+"post_attention_layernorm.weight"),nullptr,epsilon,rows);
        x=rt.add(x,rt.linear(rt.decoder_gate(n,model,p),model,p+"mlp.down_proj.weight"));
    }
    cache.length+=input.shape[0];return x;
}
Tensor decoder_logits(MetalRuntime &rt,const ModelPackage &model,const Tensor &hidden){
    NSDictionary *cfg=model.index[@"config"][@"thinker_config"][@"text_config"];
    float epsilon=[cfg[@"rms_norm_eps"] floatValue];
    if(hidden.shape[0]==1 && hybrid_decode(model.index)){
        auto [qx,xs]=rt.norm_quant(hidden,model.weight("thinker.model.norm.weight"),epsilon);
        return rt.gemv_w8a8(qx,xs,model,"thinker.lm_head.weight");
    }
    return rt.linear(rt.norm(hidden,model.weight("thinker.model.norm.weight"),nullptr,epsilon),model,"thinker.lm_head.weight");
}

// ---- Speculative verify batch (experiment) -----------------------------------
// m draft rows through the .decode INT8 weights (tensor_w8a16, f16 activations),
// KV written at the current cache slot. Greedy chain acceptance happens on the
// host; rejected rows' KV entries sit past cache.length and are overwritten by
// the next batch.
Tensor decoder_forward_decode_batch(MetalRuntime &rt,const ModelPackage &model,DecoderCache &cache,const Tensor &input){
    NSDictionary *cfg=model.index[@"config"][@"thinker_config"][@"text_config"];
    float epsilon=[cfg[@"rms_norm_eps"] floatValue],theta=[cfg[@"rope_theta"] floatValue];
    if(input.dtype!="f32" || input.shape.size()!=2 || input.shape[1]!=1024 || input.shape[0]<2)
        throw std::runtime_error("invalid verify batch input");
    if(cache.length+input.shape[0]>cache.capacity)throw std::runtime_error("decoder KV capacity exceeded");
    auto x=input;
    for(unsigned layer=0;layer<28;++layer){
        std::string p="thinker.model.layers."+std::to_string(layer)+".";
        auto n=rt.norm(x,model.weight(p+"input_layernorm.weight"),nullptr,epsilon);
        auto q=rt.linear_decode(n,model,p+"self_attn.q_proj.weight");
        auto k=rt.linear_decode(n,model,p+"self_attn.k_proj.weight");
        auto v=rt.linear_decode(n,model,p+"self_attn.v_proj.weight");
        auto rq=rt.head_norm_rope(q,model.weight(p+"self_attn.q_norm.weight"),16,unsigned(cache.length),epsilon,theta);
        auto rk=rt.head_norm_rope(k,model.weight(p+"self_attn.k_norm.weight"),8,unsigned(cache.length),epsilon,theta);
        auto kd=cache.keys[layer],vd=cache.values[layer];
        kd.offset+=cache.length*1024*4;vd.offset+=cache.length*1024*4;
        kd.shape=rk.shape;vd.shape=v.shape;
        rt.copy(rk,kd);rt.copy(v,vd);
        auto a=rt.causal_attention(rq,cache.keys[layer],cache.values[layer],unsigned(cache.length));
        x=rt.add(x,rt.linear_decode(a,model,p+"self_attn.o_proj.weight"));
        n=rt.norm(x,model.weight(p+"post_attention_layernorm.weight"),nullptr,epsilon);
        auto gate=rt.linear_decode(n,model,p+"mlp.gate_proj.weight");
        auto up=rt.linear_decode(n,model,p+"mlp.up_proj.weight");
        x=rt.add(x,rt.linear_decode(rt.swiglu(gate,up),model,p+"mlp.down_proj.weight"));
    }
    return x;
}
Tensor decoder_logits_decode_batch(MetalRuntime &rt,const ModelPackage &model,const Tensor &hidden){
    NSDictionary *cfg=model.index[@"config"][@"thinker_config"][@"text_config"];
    float epsilon=[cfg[@"rms_norm_eps"] floatValue];
    return rt.linear_decode(rt.norm(hidden,model.weight("thinker.model.norm.weight"),nullptr,epsilon),
                            model,"thinker.lm_head.weight");
}

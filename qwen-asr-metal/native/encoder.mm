#include "runtime.hpp"
#include <algorithm>
#include <stdexcept>

Tensor encode_mel(MetalRuntime &rt,const ModelPackage &model,const Tensor &mel){
    NSDictionary *audio=model.index[@"config"][@"thinker_config"][@"audio_config"];
    size_t hidden=[audio[@"d_model"] unsignedLongLongValue],layers=[audio[@"encoder_layers"] unsignedLongLongValue];
    unsigned heads=[audio[@"encoder_attention_heads"] unsignedIntValue];
    size_t chunk=2*[audio[@"n_window"] unsignedLongLongValue],window=[audio[@"n_window_infer"] unsignedLongLongValue];
    if((hidden!=896 && hidden!=1024) || (layers!=18 && layers!=24) || chunk!=100 || window!=800 || mel.shape.size()!=2 || mel.shape[0]!=128)
        throw std::runtime_error("unsupported encoder configuration");
    const std::string p="thinker.audio_tower.";
    // half_out marks producers whose only consumer is a GEMM: they emit f16
    // directly, replacing the separate cast_half pass bit-for-bit.
    auto project=[&](const Tensor &x,const std::string &name,bool gelu=false,bool half_out=false){
        auto y=rt.linear(x,model,name+".weight");return rt.bias_gelu(y,&model.weight(name+".bias"),gelu,half_out);
    };
    size_t frames=mel.shape[1],tokens=0;
    for(size_t start=0;start<frames;start+=chunk)tokens+=(std::min(chunk,frames-start)+7)/8;
    auto x=rt.allocate({tokens,hidden});size_t token_offset=0;
    for(size_t start=0;start<frames;start+=chunk){
        size_t width=std::min(chunk,frames-start),height=128;
        auto stem=rt.mel_chunk(mel,start,width);
        for(int layer=1;layer<=3;++layer){
            auto cols=rt.im2col(stem,height,width,true);
            stem=project(cols,p+"conv2d"+std::to_string(layer),true);
            height=(height+1)/2;width=(width+1)/2;
        }
        auto flat=rt.flatten_conv(stem,height,width,true);
        auto projected=rt.position(rt.linear(flat,model,p+"conv_out.weight"));
        Tensor destination=x;destination.offset+=token_offset*hidden*4;destination.shape={width,hidden};
        rt.copy(projected,destination);token_offset+=width;
    }
    unsigned window_tokens=unsigned(((chunk+7)/8)*(window/chunk));
    for(size_t layer=0;layer<layers;++layer){
        auto lp=p+"layers."+std::to_string(layer)+".";
        auto normalized=rt.norm(x,model.weight(lp+"self_attn_layer_norm.weight"),&model.weight(lp+"self_attn_layer_norm.bias"),1e-5f,true);
        Tensor q,k,v;
        if(model.has(lp+"self_attn.fused_qkv_prefill.weight")){
            auto fused=rt.linear(normalized,model,lp+"self_attn.fused_qkv_prefill.weight");
            size_t rows=fused.shape[0],width=fused.shape[1]/3;
            q=rt.allocate({rows,width});k=rt.allocate({rows,width});v=rt.allocate({rows,width});
            rt.split_qkv_bias(fused,model.weight(lp+"self_attn.fused_qkv_prefill.bias"),q,k,v);
        } else {
            q=project(normalized,lp+"self_attn.q_proj");
            k=project(normalized,lp+"self_attn.k_proj");
            v=project(normalized,lp+"self_attn.v_proj");
        }
        auto attn=rt.attention(q,k,v,heads,window_tokens,true);
        x=rt.add(x,project(attn,lp+"self_attn.out_proj"));
        normalized=rt.norm(x,model.weight(lp+"final_layer_norm.weight"),&model.weight(lp+"final_layer_norm.bias"),1e-5f,true);
        auto ffn=project(normalized,lp+"fc1",true,true);
        x=rt.add(x,project(ffn,lp+"fc2"));
    }
    x=rt.norm(x,model.weight(p+"ln_post.weight"),&model.weight(p+"ln_post.bias"),1e-5f,true);
    return project(project(x,p+"proj1",true,true),p+"proj2");
}

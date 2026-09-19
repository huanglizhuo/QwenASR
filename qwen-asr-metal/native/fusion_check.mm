#include "runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
static void compare(const Tensor &a,const Tensor &b){
    if(a.shape!=b.shape)throw std::runtime_error("fusion output shape mismatch");
    const float *x=(const float *)((const char *)a.buffer.contents+a.offset),*y=(const float *)((const char *)b.buffer.contents+b.offset);
    for(size_t i=0;i<a.elements();++i)if(!std::isfinite(x[i]) || !std::isfinite(y[i]) || std::abs(x[i]-y[i])>1e-5f*(1+std::abs(x[i])))
        throw std::runtime_error("fusion differs at element "+std::to_string(i));
}
int main(int argc,const char **argv){@autoreleasepool{try{
    if(argc!=3)throw std::runtime_error("usage: fusion-check <hybrid-package> <kernels>");
    MetalRuntime rt(argv[2]);ModelPackage model(rt.device,argv[1]);
    if(![model.index[@"precision"] isEqual:@"hybrid"])throw std::runtime_error("hybrid package required");
    for(unsigned layer:{0,13,27})for(float magnitude:{.001f,1.0f,100.0f}){
        std::vector<float> data(1024);for(size_t i=0;i<data.size();++i)data[i]=float(int((i*17+13)%37)-18)*magnitude/32;
        auto x=rt.upload(data.data(),1,1024);std::string p="thinker.model.layers."+std::to_string(layer)+".";
        rt.begin();std::vector<Tensor> reference;
        for(const char *name:{"q_proj","k_proj","v_proj"})reference.push_back(rt.linear(x,model,p+"self_attn."+name+".weight"));
        auto qkv=rt.decoder_qkv(x,model,p);
        auto g=rt.linear(x,model,p+"mlp.gate_proj.weight"),u=rt.linear(x,model,p+"mlp.up_proj.weight");
        auto separate=rt.swiglu(g,u),fused=rt.decoder_gate(x,model,p);rt.finish();
        for(size_t i=0;i<3;++i)compare(reference[i],qkv[i]);compare(separate,fused);
    }
    puts("QKV and SwiGLU fusion regression passed: 3 layers x 3 activation magnitudes");return 0;
}catch(const std::exception &e){fprintf(stderr,"fusion check failed: %s\n",e.what());return 2;}}}

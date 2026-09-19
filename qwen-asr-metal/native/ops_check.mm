#include "runtime.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <limits>

int main(int argc,const char **argv){
    @autoreleasepool{
        try{
            if(argc!=2)throw std::runtime_error("usage: ops-check <kernels-dir>");
            MetalRuntime rt(argv[1]);
            const float values[]={-100,-20,-12,-10,-5,-1,0,1,5,10,12,20,100};
            auto x=rt.upload(values,1,sizeof(values)/sizeof(float));
            rt.begin();auto fused=rt.bias_gelu(x,nullptr,true);auto separate=rt.activation(x,0);rt.finish();
            for(const Tensor *output:{&fused,&separate})for(size_t i=0;i<x.elements();++i){
                double v=values[i],expected=.5*v*(1+std::tanh(.7978845608*(v+.044715*v*v*v)));
                float actual=((float *)output->buffer.contents)[i];
                if(!std::isfinite(actual) || std::abs(actual-expected)>2e-5)
                    throw std::runtime_error("GELU regression at input "+std::to_string(v)+", actual="+std::to_string(actual));
            }
            puts("GELU extreme-activation regression passed (fused and standalone)");
            for(size_t width:{5000,151936}){
                std::vector<float> logits(4*width,-5.0f);
                logits[1023]=logits[width-1]=10; // tie across partitions: choose smallest ID
                logits[width+width-1]=20; // partial final tile
                logits[2*width+1024]=std::numeric_limits<float>::quiet_NaN();
                auto input=rt.upload(logits.data(),4,width);rt.begin();auto top=rt.argmax(input);rt.finish();
                const int32_t *ids=(const int32_t *)top.buffer.contents;
                if(ids[0]!=1023 || ids[1]!=int32_t(width-1) || ids[2]!=-1 || ids[3]!=0)
                    throw std::runtime_error("parallel argmax tie/tail/NaN regression");
            }
            puts("Parallel argmax regression passed (5000/151936 classes, ties, tails, NaN, negative logits)");return 0;
        }catch(const std::exception &e){fprintf(stderr,"ops check failed: %s\n",e.what());return 2;}
    }
}

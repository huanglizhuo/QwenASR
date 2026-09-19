#include "runtime.hpp"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

int main(int argc,const char **argv){
    @autoreleasepool {
        try{
            if(argc!=6)throw std::runtime_error("usage: encoder-check <package> <kernels-dir> <mel.f32> <frames> <output.f32>");
            MetalRuntime rt(argv[2]);ModelPackage model(rt.device,argv[1]);
            size_t frames=std::stoul(argv[4]);
            if(frames==0 || frames>3000)throw std::runtime_error("encoder validation supports 1..3000 frames");
            std::vector<float> mel(128*frames);
            std::ifstream input(argv[3],std::ios::binary);
            input.read((char *)mel.data(),mel.size()*4);
            if(!input || input.peek()!=EOF)throw std::runtime_error("mel file length mismatch");
            auto x=rt.upload(mel.data(),128,frames);
            Tensor encoded;
            double gpu_ms=0,wall_ms=0;
            for(int iteration=0;iteration<3;++iteration){
            auto start=std::chrono::steady_clock::now();
            rt.begin();encoded=encode_mel(rt,model,x);gpu_ms=rt.finish();
            const float *values=(const float *)((const char *)encoded.buffer.contents+encoded.offset);
            for(size_t i=0;i<encoded.elements();++i) if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite encoder output at element "+std::to_string(i));
            wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            fprintf(stderr,"encoder iteration %d: GPU %.3f ms, host %.3f ms\n",iteration,gpu_ms,wall_ms);
            }
            std::ofstream out(argv[5],std::ios::binary);
            out.write((const char *)encoded.buffer.contents+encoded.offset,encoded.elements()*4);
            if(!out)throw std::runtime_error("cannot write encoder output");
            printf("{\"gpu_encoder_ms\":%.6f,\"host_encoder_ms\":%.6f,\"warmups\":2,\"tokens\":%zu,\"hidden\":%zu}\n",gpu_ms,wall_ms,encoded.shape[0],encoded.shape[1]);
            return 0;
        }catch(const std::exception &e){fprintf(stderr,"encoder check failed: %s\n",e.what());return 2;}
    }
}

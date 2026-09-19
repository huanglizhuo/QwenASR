#include "runtime.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
int main(int argc,const char **argv){@autoreleasepool{try{
    if(argc!=5)throw std::runtime_error("usage: frontend-check <kernels> <pcm.f32> <samples> <mel.f32>");
    size_t count=std::stoul(argv[3]);if(count<160 || count>480000)throw std::runtime_error("invalid sample count");
    std::vector<float> pcm(count);std::ifstream f(argv[2],std::ios::binary);f.read((char *)pcm.data(),count*4);
    if(!f || f.peek()!=EOF)throw std::runtime_error("invalid PCM length");
    MetalRuntime rt(argv[1]);Tensor mel;double gpu_ms=0,host_ms=0;
    for(unsigned iteration=0;iteration<3;++iteration){
        auto start=std::chrono::steady_clock::now();auto input=rt.upload(pcm.data(),1,count);
        rt.begin();mel=rt.mel_spectrogram(input);gpu_ms=rt.finish();
        host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    auto values=(const float *)mel.buffer.contents;
    for(size_t i=0;i<mel.elements();++i)if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite Mel");
    std::ofstream out(argv[4],std::ios::binary);out.write((const char *)values,mel.elements()*4);
    if(!out)throw std::runtime_error("cannot write Mel");
    printf("{\"gpu_ms\":%.6f,\"host_ms\":%.6f,\"frames\":%zu}\n",gpu_ms,host_ms,mel.shape[1]);return 0;
}catch(const std::exception &e){fprintf(stderr,"frontend check failed: %s\n",e.what());return 2;}}}

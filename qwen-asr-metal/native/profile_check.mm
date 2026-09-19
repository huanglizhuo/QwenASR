#include "runtime.hpp"
#include <cstdio>
#include <fstream>
#include <stdexcept>
int main(int argc,const char **argv){@autoreleasepool{try{
    if(argc!=6)throw std::runtime_error("usage: profile-check <package> <kernels> <input.f32> <seq> <profile.jsonl>");
    size_t seq=std::stoul(argv[4]);if(seq<2 || seq>2048)throw std::runtime_error("invalid sequence length");
    std::vector<float> input(seq*1024);std::ifstream f(argv[3],std::ios::binary);f.read((char *)input.data(),input.size()*4);
    if(!f || f.peek()!=EOF)throw std::runtime_error("invalid reference input");
    MetalRuntime rt(argv[2]);ModelPackage model(rt.device,argv[1]);DecoderCache cache(rt,seq+16);
    auto x=rt.upload(input.data(),seq,1024),prefill=x;prefill.shape[0]=seq-1;
    rt.begin();decoder_forward(rt,model,cache,prefill);rt.finish();
    auto last=x;last.offset+=(seq-1)*1024*4;last.shape[0]=1;
    rt.enable_profiling(argv[5]);
    for(int i=0;i<4;++i){cache.length=seq-1;rt.begin();auto h=decoder_forward(rt,model,cache,last);rt.argmax(decoder_logits(rt,model,h));rt.finish();}
    puts("Development GPU counters recorded; these timings are not end-to-end acceptance results.");return 0;
}catch(const std::exception &e){fprintf(stderr,"profile check failed: %s\n",e.what());return 2;}}}

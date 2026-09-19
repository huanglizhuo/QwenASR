#include "runtime.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
static void save(const std::string &path,const Tensor &t){
    const float *p=(const float *)((const char *)t.buffer.contents+t.offset);
    for(size_t i=0;i<t.elements();++i)if(!std::isfinite(p[i]))throw std::runtime_error("nonfinite decoder output at "+std::to_string(i));
    std::ofstream out(path,std::ios::binary);out.write((const char *)p,t.elements()*4);
    if(!out)throw std::runtime_error("cannot write decoder output");
}
int main(int argc,const char **argv){@autoreleasepool{try{
    if(argc!=6)throw std::runtime_error("usage: decoder-check <package> <kernels> <reference-prefix> <output-prefix> <full|split>");
    bool split=std::string(argv[5])=="split";
    if(!split && std::string(argv[5])!="full")throw std::runtime_error("invalid check mode");
    NSError *error=nil;
    NSDictionary *metadata=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@((std::string(argv[3])+".json").c_str())] options:0 error:&error];
    size_t seq=[metadata[@"seq"] unsignedLongLongValue];
    if(seq<2 || seq>4096)throw std::runtime_error("invalid reference length");
    std::vector<int32_t> rows;for(NSNumber *r in metadata[@"rows"])rows.push_back(r.intValue);
    std::vector<float> input(seq*1024);std::ifstream f(std::string(argv[3])+".input.f32",std::ios::binary);
    f.read((char *)input.data(),input.size()*4);if(!f || f.peek()!=EOF)throw std::runtime_error("invalid input length");
    MetalRuntime rt(argv[2]);ModelPackage model(rt.device,argv[1]);DecoderCache cache(rt,seq+16);
    auto x=rt.upload(input.data(),seq,1024);Tensor hidden,logits,top;
    double gpu_ms=0,host_ms=0;
    for(unsigned iteration=0;iteration<3;++iteration){
        auto start=std::chrono::steady_clock::now();cache.reset();rt.begin();
        if(!split)hidden=decoder_forward(rt,model,cache,x);
        else{
            // Exercise both multi-token prefill and one-token GEMV, including
            // reading KV from a prior command buffer at a nonzero position.
            auto first=x;first.shape[0]=seq-1;
            auto before=decoder_forward(rt,model,cache,first);gpu_ms=rt.finish();rt.begin();
            auto last=x;last.offset+=(seq-1)*1024*4;last.shape[0]=1;
            auto after=decoder_forward(rt,model,cache,last);
            hidden=rt.allocate(x.shape);auto hd=hidden;hd.shape=before.shape;rt.copy(before,hd);
            hd.offset+=(seq-1)*1024*4;hd.shape=after.shape;rt.copy(after,hd);
        }
        logits=decoder_logits(rt,model,rt.gather_rows(hidden,rows));top=rt.argmax(logits);
        double elapsed=rt.finish();gpu_ms=split?gpu_ms+elapsed:elapsed;
        host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        fprintf(stderr,"decoder iteration %u: GPU %.3f ms, host %.3f ms\n",iteration,gpu_ms,host_ms);
    }
    save(std::string(argv[4])+".hidden.f32",hidden);save(std::string(argv[4])+".logits.f32",logits);
    const int32_t *ids=(const int32_t *)top.buffer.contents;
    printf("{\"gpu_ms\":%.6f,\"host_ms\":%.6f,\"seq\":%zu,\"argmax\":[",gpu_ms,host_ms,seq);
    for(size_t i=0;i<rows.size();++i){if(ids[i]<0)throw std::runtime_error("invalid GPU argmax");printf("%s%d",i?",":"",ids[i]);}
    printf("]}\n");return 0;
}catch(const std::exception &e){fprintf(stderr,"decoder check failed: %s\n",e.what());return 2;}}}

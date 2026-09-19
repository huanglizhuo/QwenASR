#include "runtime.hpp"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

int main(int argc,const char **argv){
    @autoreleasepool {
        try{
            if(argc!=6)throw std::runtime_error("usage: runtime-check <package> <kernels-dir> <weight-name> <rows> <output-prefix>");
            MetalRuntime rt(argv[2]);ModelPackage model(rt.device,argv[1]);
            auto &weight=model.weight(argv[3]);
            size_t k=weight.elements()/weight.shape[0],rows=std::stoul(argv[4]);
            if(rows==0 || rows>1024)throw std::runtime_error("rows outside validation range");
            std::string name=argv[3];
            bool hybrid=rows==1 && [model.index[@"precision"] isEqual:@"hybrid"] && (name.rfind("thinker.model.layers.",0)==0 || name=="thinker.lm_head.weight");
            std::vector<float> values(rows*k);
            for(size_t i=0;i<values.size();++i)values[i]=float(int((i*17+13)%37)-18)/32;
            auto input=rt.upload(values.data(),rows,k);
            rt.begin();auto output=rt.linear(input,model,argv[3]);
            double ms=rt.finish();
            std::string prefix=argv[5];
            std::ofstream file(prefix+".f32",std::ios::binary);
            file.write((const char *)output.buffer.contents+output.offset,output.elements()*4);
            if(!file)throw std::runtime_error("cannot write matrix output");
            // Also exercise norms, activation and residual in one command, with
            // no intermediate host synchronization or numerical computation.
            const auto &norm=model.weight("thinker.model.layers.0.input_layernorm.weight");
            std::vector<float> normValues(1024);
            for(size_t i=0;i<normValues.size();++i)normValues[i]=float(int((i*17+13)%37)-18)/32;
            auto normInput=rt.upload(normValues.data(),1,1024);
            rt.begin();auto normalized=rt.norm(normInput,norm,nullptr,1e-6f);
            auto activated=rt.activation(normalized,0);auto residual=rt.add(normInput,activated);
            rt.finish();
            std::ofstream normFile(prefix+".norm.f32",std::ios::binary);
            normFile.write((const char *)residual.buffer.contents,1024*4);
            if(!normFile)throw std::runtime_error("cannot write norm output");
            printf("{\"device\":\"%s\",\"rows\":%zu,\"columns\":%zu,\"gpu_linear_ms\":%.6f,\"precision\":\"%s\"}\n",
                   rt.device.name.UTF8String,rows,weight.shape[0],ms,hybrid?"w8a8":weight.dtype.c_str());
            return 0;
        }catch(const std::exception &e){fprintf(stderr,"runtime check failed: %s\n",e.what());return 2;}
    }
}

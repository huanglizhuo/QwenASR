#include "runtime.hpp"
#include "bridge.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
struct Engine {
    MetalRuntime runtime;
    ModelPackage model;
    std::unique_ptr<DecoderCache> cache;
    bool align;
    Engine(const char *package,const char *kernels):runtime(kernels),model(runtime.device,package){
        NSString *task=model.index[@"task"];
        if(![task isEqual:@"asr"] && ![task isEqual:@"align"])throw std::runtime_error("invalid package task");
        align=[task isEqual:@"align"];
    }
};
static void failure(char *buffer,size_t n,const char *message){if(buffer && n)snprintf(buffer,n,"%s",message);}
extern "C" void *qmetal_create(const char *package,const char *kernels,char *error,size_t capacity){@autoreleasepool{try{
    if(!package || !kernels)throw std::runtime_error("missing package/kernel path");
    return new Engine(package,kernels);
}catch(const std::exception &e){failure(error,capacity,e.what());return nullptr;}}}
extern "C" void qmetal_destroy(void *handle){@autoreleasepool{delete static_cast<Engine *>(handle);}}
extern "C" int qmetal_infer(void *handle,const float *pcm,size_t samples,
    const int32_t *prefix,size_t prefix_count,const int32_t *suffix,size_t suffix_count,
    const int32_t *timestamp_rows,size_t row_count,uint32_t max_new_tokens,
    int32_t *output,size_t output_capacity,QMetalStats *stats,char *error,size_t error_capacity){@autoreleasepool{try{
    if(!handle || !pcm || !prefix || !suffix || !prefix_count || !suffix_count || !output || !stats)
        throw std::runtime_error("invalid inference arguments");
    if(samples<160 || samples>480000 || prefix_count>4096 || suffix_count>4096 || max_new_tokens>2048)
        throw std::runtime_error("input exceeds v1 single-clip limits");
    for(size_t i=0;i<samples;++i)if(!std::isfinite(pcm[i]))throw std::runtime_error("nonfinite PCM");
    auto &e=*static_cast<Engine *>(handle);auto &rt=e.runtime;*stats={};
    if((e.align && (!timestamp_rows || !row_count || row_count>output_capacity)) || (!e.align && (max_new_tokens==0 || max_new_tokens>output_capacity)))
        throw std::runtime_error("invalid output capacity");
    size_t frames=samples/160,audio_tokens=0;
    for(size_t i=0;i<frames;i+=100)audio_tokens+=(std::min(size_t(100),frames-i)+7)/8;
    size_t total=prefix_count+audio_tokens+suffix_count,capacity=total+(e.align?0:max_new_tokens);
    if(capacity>65536)throw std::runtime_error("sequence too long");
    if(!e.cache || e.cache->capacity<capacity)e.cache=std::make_unique<DecoderCache>(rt,capacity);
    e.cache->reset();
    auto input=rt.upload(pcm,1,samples);rt.begin();
    auto encoded=encode_mel(rt,e.model,rt.mel_spectrogram(input));
    auto before=rt.embedding(e.model,std::vector<int32_t>(prefix,prefix+prefix_count));
    auto after=rt.embedding(e.model,std::vector<int32_t>(suffix,suffix+suffix_count));
    auto prompt=rt.allocate({total,1024});auto dest=prompt;
    dest.shape=before.shape;rt.copy(before,dest);
    dest.offset=prefix_count*1024*4;dest.shape=encoded.shape;rt.copy(encoded,dest);
    dest.offset=(prefix_count+audio_tokens)*1024*4;dest.shape=after.shape;rt.copy(after,dest);
    Tensor hidden;
    if(e.align)hidden=decoder_forward(rt,e.model,*e.cache,prompt);
    else{
        // Match the Rust split exactly: the final prompt token takes the same
        // single-token precision policy as every generated token.
        auto prefill=prompt;prefill.shape[0]=total-1;
        decoder_forward(rt,e.model,*e.cache,prefill);
        auto last=prompt;last.offset+=(total-1)*1024*4;last.shape[0]=1;
        hidden=decoder_forward(rt,e.model,*e.cache,last);
    }
    stats->audio_tokens=uint32_t(audio_tokens);
    if(e.align){
        std::vector<int32_t> positions;
        for(size_t i=0;i<row_count;++i){
            if(timestamp_rows[i]<0 || size_t(timestamp_rows[i])>=suffix_count)throw std::runtime_error("timestamp position outside suffix");
            positions.push_back(int32_t(prefix_count+audio_tokens)+timestamp_rows[i]);
        }
        auto ids=rt.argmax(decoder_logits(rt,e.model,rt.gather_rows(hidden,positions)));
        stats->gpu_ms=rt.finish();const int32_t *values=(const int32_t *)ids.buffer.contents;
        for(size_t i=0;i<row_count;++i){if(values[i]<0)throw std::runtime_error("nonfinite alignment logits");output[i]=values[i];}
        stats->decoder_positions=uint32_t(e.cache->length);stats->generated=uint32_t(row_count);return int(row_count);
    }
    // Speculative verify batch width (experiment): >1 switches generation to
    // batched W8A16 verify with a self-chained draft; 1 keeps the single-token
    // W8A8 path. Outputs are greedy of the W8A16-verify quantization either way.
    const char *spec_env=getenv("QWEN_SPEC");
    unsigned spec=spec_env?unsigned(std::stoul(spec_env)):1u;
    if(spec>1u){
        if(spec>8u)spec=8u;
        if(spec>max_new_tokens)spec=max_new_tokens;
        // Complete the pending prefill+first-position logits, then run the
        // generation loop as batched verifies seeded with a self-chained draft.
        auto first=rt.argmax(decoder_logits(rt,e.model,hidden));
        stats->gpu_ms=rt.finish();
        int32_t current=*(int32_t *)first.buffer.contents;
        if(current<0)throw std::runtime_error("nonfinite ASR logits");
        output[0]=current;stats->generated=1;
        if(current==151645 || current==151643){stats->decoder_positions=uint32_t(e.cache->length);return 1;}
        std::vector<int32_t> draft(spec,current);
        uint32_t produced=1;
        while(produced<max_new_tokens){
            draft[0]=current;
            rt.begin();
            auto embed=rt.embedding(e.model,draft);
            auto batch=decoder_forward_decode_batch(rt,e.model,*e.cache,embed);
            auto row_ids=rt.argmax(decoder_logits_decode_batch(rt,e.model,batch));
            stats->gpu_ms+=rt.finish();
            const int32_t *values=(const int32_t *)row_ids.buffer.contents;
            for(unsigned r=0;r<spec;++r)if(values[r]<0)throw std::runtime_error("nonfinite ASR logits");
            // ids[j] is greedy-next after draft[0..j]; accept while the draft
            // chain stays on the greedy path.
            unsigned count=1;
            while(count<spec && draft[count]==values[count-1])++count;
            for(unsigned j=0;j<count && produced+j<max_new_tokens;++j){
                output[produced+j]=values[j];
                if(values[j]==151645 || values[j]==151643){
                    stats->generated=produced+j+1;
                    e.cache->length+=j+1;
                    stats->decoder_positions=uint32_t(e.cache->length);
                    return int(produced+j+1);
                }
            }
            produced+=count;
            e.cache->length+=count;
            if(getenv("QWEN_SPEC_DEBUG"))fprintf(stderr,"spec batch: accepted=%u/%u produced=%u gpu=%.2f\n",count,spec,produced,stats->gpu_ms);
            if(produced>=max_new_tokens)break;
            // Next draft: the stale self-chain from this batch's predictions.
            current=values[count-1];
            draft.assign(values+(count-1),values+spec);
            draft.resize(spec,values[spec-1]);
        }
        throw std::runtime_error("ASR exceeded token limit before EOS");
    }
    auto last=hidden;
    auto ids=rt.argmax(decoder_logits(rt,e.model,last));stats->gpu_ms=rt.finish();
    for(uint32_t i=0;i<max_new_tokens;++i){
        int32_t token=*(int32_t *)ids.buffer.contents;
        if(token<0)throw std::runtime_error("nonfinite ASR logits");
        output[i]=token;stats->generated=i+1;
        if(token==151645 || token==151643){stats->decoder_positions=uint32_t(e.cache->length);return int(i+1);}
        if(i+1==max_new_tokens)throw std::runtime_error("ASR exceeded token limit before EOS");
        rt.begin();auto embed=rt.embedding(e.model,{token});
        auto step=decoder_forward(rt,e.model,*e.cache,embed);ids=rt.argmax(decoder_logits(rt,e.model,step));stats->gpu_ms+=rt.finish();
    }
    throw std::runtime_error("unreachable decoder exit");
}catch(const std::exception &e){failure(error,error_capacity,e.what());return -1;}}}

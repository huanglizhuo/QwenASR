// Concatenated after quantized.metal by the runtime.
inline float stable_gelu(float v) {
    // The fast Metal tanh lowering can overflow internally for the cubic
    // argument at v=12. Beyond |v|=10, tanh has already rounded to +/-1 in
    // FP32; these tails preserve that result without evaluating an overflow.
    if(v>=10.0f)return v;
    if(v<=-10.0f)return 0.0f;
    return 0.5f*v*(1.0f+tanh(0.7978845608f*(v+0.044715f*v*v*v)));
}
kernel void cast_half(device float *x [[buffer(0)]], device half *y [[buffer(1)]],
                      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i<count) y[i]=half(x[i]);
}

kernel void tensor_f16(device half *x [[buffer(0)]], device half *w [[buffer(1)]],
                       device float *out [[buffer(2)]], constant Shape &s [[buffer(4)]],
                       uint2 group [[threadgroup_position_in_grid]]) {
    auto X=tensor(x,dextents<int,2>{int(s.k),int(s.m)},array<int,2>{1,int(s.k)});
    auto W=tensor(w,dextents<int,2>{int(s.k),int(s.n)},array<int,2>{1,int(s.k)});
    auto O=tensor(out,dextents<int,2>{int(s.n),int(s.m)},array<int,2>{1,int(s.n)});
    auto a=X.slice(0,group.y*32);
    auto b=W.slice(0,group.x*32);
    auto c=O.slice(group.x*32,group.y*32);
    constexpr auto desc=matmul2d_descriptor(32,32,int(dynamic_extent),false,true,false);
    matmul2d<desc,execution_simdgroup> op;
    auto result=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,result); result.store(c);
}

struct NormArgs { uint width; float epsilon; uint layer_norm; };
kernel void norm_f32(device float *x [[buffer(0)]], device float *weight [[buffer(1)]],
                     device float *bias [[buffer(2)]], device float *y [[buffer(3)]],
                     constant NormArgs &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
    // A single SIMD group owns each row. FP32 reduction never spills a partial
    // across threadgroups, including head RMSNorm and full hidden LayerNorm.
    float sum=0, squares=0;
    for(uint j=lane;j<p.width;j+=32) { float v=x[row*p.width+j]; sum+=v; squares+=v*v; }
    float mean=p.layer_norm ? simd_sum(sum)/p.width : 0;
    float variance;
    if(p.layer_norm) {
        float centered=0;
        for(uint j=lane;j<p.width;j+=32) {float v=x[row*p.width+j]-mean; centered+=v*v;}
        variance=simd_sum(centered)/p.width;
    } else variance=simd_sum(squares)/p.width;
    float inv=rsqrt(variance+p.epsilon);
    for(uint j=lane;j<p.width;j+=32)
        y[row*p.width+j]=(x[row*p.width+j]-mean)*inv*weight[j]+(p.layer_norm?bias[j]:0.0f);
}

struct ElementArgs { uint count, operation; };
kernel void activate_f32(device float *x [[buffer(0)]], device float *y [[buffer(1)]],
                         constant ElementArgs &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.count) return;
    float v=x[i];
    y[i]=p.operation==0 ? stable_gelu(v) : v/(1.0f+exp(-v));
}

kernel void add_f32(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                    device float *out [[buffer(2)]], constant uint &count [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if(i<count) out[i]=a[i]+b[i];
}

struct BiasArgs {uint count,width,has_bias,gelu;};
kernel void bias_gelu_f32(device float *x [[buffer(0)]],device float *bias [[buffer(1)]],
                          device float *y [[buffer(2)]],constant BiasArgs &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    float v=x[i]+(p.has_bias?bias[i%p.width]:0.0f);
    y[i]=p.gelu?stable_gelu(v):v;
}
struct ChunkArgs {uint frames,start,width;};
kernel void mel_chunk_f32(device float *mel [[buffer(0)]],device float *out [[buffer(1)]],
                          constant ChunkArgs &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<128*p.width)out[i]=mel[(i/p.width)*p.frames+p.start+i%p.width];
}
struct ConvArgs {uint height,width,channels,out_height,out_width;};
kernel void im2col_f32(device float *x [[buffer(0)]],device float *out [[buffer(1)]],
                       constant ConvArgs &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint k=p.channels*9,total=p.out_height*p.out_width*k;
    if(i>=total)return;
    uint row=i/k,col=i%k,c=col/9;
    int h=int((row/p.out_width)*2+(col%9)/3)-1;
    int w=int((row%p.out_width)*2+col%3)-1;
    out[i]=(h>=0 && w>=0 && h<int(p.height) && w<int(p.width))?x[(h*p.width+w)*p.channels+c]:0;
}
kernel void flatten_conv_f32(device float *x [[buffer(0)]],device float *out [[buffer(1)]],
                             constant ConvArgs &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.width*p.height*p.channels)return;
    uint t=i/(p.height*p.channels), col=i%(p.height*p.channels);
    uint c=col/p.height,f=col%p.height;
    out[i]=x[(f*p.width+t)*p.channels+c];
}
kernel void position_f32(device float *x [[buffer(0)]],device float *out [[buffer(1)]],
                         constant uint2 &shape [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=shape.x*shape.y)return;
    uint d=i%shape.y,half_dim=shape.y/2;
    float angle=float(i/shape.y)*exp(-float(d%half_dim)*log(10000.0f)/float(half_dim-1));
    out[i]=x[i]+(d<half_dim?sin(angle):cos(angle));
}
struct AttentionArgs {uint seq,heads,dim,window;};
kernel void window_attention_f32(device float *q [[buffer(0)]],device float *k [[buffer(1)]],
                                 device float *v [[buffer(2)]],device float *out [[buffer(3)]],
                                 constant AttentionArgs &p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_threadgroup]]) {
    uint query=group.x,head=group.y,stride=p.heads*p.dim;
    uint begin=(query/p.window)*p.window,end=min(begin+p.window,p.seq);
    float qv[4]={0,0,0,0},acc[4]={0,0,0,0};
    for(uint j=0;j<p.dim/32;++j)qv[j]=q[query*stride+head*p.dim+lane+j*32];
    float maximum=-INFINITY,denom=0;
    for(uint t=begin;t<end;++t) {
        float dot=0;
        for(uint j=0;j<p.dim/32;++j)dot+=qv[j]*k[t*stride+head*p.dim+lane+j*32];
        float score=simd_sum(dot)*rsqrt(float(p.dim));
        float next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        denom=denom*old+prob;maximum=next;
        for(uint j=0;j<p.dim/32;++j)acc[j]=acc[j]*old+prob*v[t*stride+head*p.dim+lane+j*32];
    }
    for(uint j=0;j<p.dim/32;++j)out[query*stride+head*p.dim+lane+j*32]=acc[j]/denom;
}
kernel void copy_f32(device float *x [[buffer(0)]],device float *out [[buffer(1)]],
                     constant uint &count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<count)out[i]=x[i];
}

// ---- Fused-prefill split kernels ---------------------------------------------
// The fused qkv GEMM writes {rows, 3*width}; downstream kernels need contiguous
// {rows, width} tensors, so these splits copy column blocks row by row. Pure
// elementwise copies (plus the same bias add bias_gelu performed) — exact.

struct SplitQKVArgs {uint rows,src_width,dst_width,has_bias;};
kernel void split_qkv_bias_f32(device float *src [[buffer(0)]],device float *bias [[buffer(1)]],
                               device float *q [[buffer(2)]],device float *k [[buffer(3)]],
                               device float *v [[buffer(4)]],constant SplitQKVArgs &p [[buffer(5)]],
                               uint i [[thread_position_in_grid]]){
    // Equal thirds (encoder): q/k/v each dst_width wide.
    if(i>=p.rows*p.dst_width)return;
    uint row=i/p.dst_width,col=i%p.dst_width;
    float value=src[row*p.src_width+col]+(p.has_bias?bias[col]:0.0f);
    q[i]=value;
    k[i]=src[row*p.src_width+p.dst_width+col]+(p.has_bias?bias[p.dst_width+col]:0.0f);
    v[i]=src[row*p.src_width+2*p.dst_width+col]+(p.has_bias?bias[2*p.dst_width+col]:0.0f);
}

// ---- Half-emitting variants -------------------------------------------------
// Each computes the identical float value as its f32 sibling and stores half(v)
// in one step: the same round-to-nearest bits cast_half produced, minus one
// full-resolution round trip through memory.

kernel void norm_f32_h(device float *x [[buffer(0)]],device float *weight [[buffer(1)]],
                       device float *bias [[buffer(2)]],device half *y [[buffer(3)]],
                       constant NormArgs &p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    float sum=0, squares=0;
    for(uint j=lane;j<p.width;j+=32) { float v=x[row*p.width+j]; sum+=v; squares+=v*v; }
    float mean=p.layer_norm ? simd_sum(sum)/p.width : 0;
    float variance;
    if(p.layer_norm) {
        float centered=0;
        for(uint j=lane;j<p.width;j+=32) {float v=x[row*p.width+j]-mean; centered+=v*v;}
        variance=simd_sum(centered)/p.width;
    } else variance=simd_sum(squares)/p.width;
    float inv=rsqrt(variance+p.epsilon);
    for(uint j=lane;j<p.width;j+=32)
        y[row*p.width+j]=half((x[row*p.width+j]-mean)*inv*weight[j]+(p.layer_norm?bias[j]:0.0f));
}
kernel void bias_gelu_f32_h(device float *x [[buffer(0)]],device float *bias [[buffer(1)]],
                            device half *y [[buffer(2)]],constant BiasArgs &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    float v=x[i]+(p.has_bias?bias[i%p.width]:0.0f);
    y[i]=p.gelu?half(stable_gelu(v)):half(v);
}
kernel void position_f32_h(device float *x [[buffer(0)]],device half *out [[buffer(1)]],
                           constant uint2 &shape [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=shape.x*shape.y)return;
    uint d=i%shape.y,half_dim=shape.y/2;
    float angle=float(i/shape.y)*exp(-float(d%half_dim)*log(10000.0f)/float(half_dim-1));
    out[i]=half(x[i]+(d<half_dim?sin(angle):cos(angle)));
}
kernel void window_attention_f32_h(device float *q [[buffer(0)]],device float *k [[buffer(1)]],
                                   device float *v [[buffer(2)]],device half *out [[buffer(3)]],
                                   constant AttentionArgs &p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
    uint query=group.x,head=group.y,stride=p.heads*p.dim;
    uint begin=(query/p.window)*p.window,end=min(begin+p.window,p.seq);
    float qv[4]={0,0,0,0},acc[4]={0,0,0,0};
    for(uint j=0;j<p.dim/32;++j)qv[j]=q[query*stride+head*p.dim+lane+j*32];
    float maximum=-INFINITY,denom=0;
    for(uint t=begin;t<end;++t) {
        float dot=0;
        for(uint j=0;j<p.dim/32;++j)dot+=qv[j]*k[t*stride+head*p.dim+lane+j*32];
        float score=simd_sum(dot)*rsqrt(float(p.dim));
        float next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        denom=denom*old+prob;maximum=next;
        for(uint j=0;j<p.dim/32;++j)acc[j]=acc[j]*old+prob*v[t*stride+head*p.dim+lane+j*32];
    }
    for(uint j=0;j<p.dim/32;++j)out[query*stride+head*p.dim+lane+j*32]=half(acc[j]/denom);
}
kernel void im2col_f32_h(device float *x [[buffer(0)]],device half *out [[buffer(1)]],
                         constant ConvArgs &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint k=p.channels*9,total=p.out_height*p.out_width*k;
    if(i>=total)return;
    uint row=i/k,col=i%k,c=col/9;
    int h=int((row/p.out_width)*2+(col%9)/3)-1;
    int w=int((row%p.out_width)*2+col%3)-1;
    out[i]=half((h>=0 && w>=0 && h<int(p.height) && w<int(p.width))?x[(h*p.width+w)*p.channels+c]:0.0f);
}
kernel void flatten_conv_f32_h(device float *x [[buffer(0)]],device half *out [[buffer(1)]],
                               constant ConvArgs &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.width*p.height*p.channels)return;
    uint t=i/(p.height*p.channels), col=i%(p.height*p.channels);
    uint c=col/p.height,f=col%p.height;
    out[i]=half(x[(f*p.width+t)*p.channels+c]);
}

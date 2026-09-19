// Concatenated after the shared kernels; all activations/KV remain on GPU.
kernel void gemv_f16(device half *x [[buffer(0)]],device half *w [[buffer(1)]],
                     device float *out [[buffer(2)]],constant Shape &s [[buffer(4)]],
                     uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint row=group.x*8+tid/32,lane=tid%32;
    if(row>=s.n)return;
    float sum=0;
    for(uint k=lane;k<s.k;k+=32)sum+=float(x[group.y*s.k+k])*float(w[row*s.k+k]);
    sum=simd_sum(sum);
    if(lane==0)out[group.y*s.n+row]=sum;
}
struct EmbedArgs {uint rows,width,quantized;};
kernel void embedding_f32(device uchar *weights [[buffer(0)]],device float *scales [[buffer(1)]],
                          device int *tokens [[buffer(2)]],device float *out [[buffer(3)]],
                          constant EmbedArgs &p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.width)return;
    uint token=uint(tokens[i/p.width]),index=token*p.width+i%p.width;
    out[i]=p.quantized?float(((device int8_t *)weights)[index])*scales[token]:float(((device half *)weights)[index]);
}
struct RopeArgs {uint heads,start;float epsilon,theta;};
kernel void head_norm_rope_f32(device float *x [[buffer(0)]],device float *weight [[buffer(1)]],
                               device float *out [[buffer(2)]],constant RopeArgs &p [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    // NeoX half rotation: pair dimensions d and d+64. All position axes are
    // identical for this audio/text sequence, matching the CPU's 1-D RoPE.
    float squares=0;
    for(uint d=lane;d<128;d+=32){float v=x[row*128+d];squares+=v*v;}
    float inv=rsqrt(simd_sum(squares)/128.0f+p.epsilon);
    uint position=p.start+row/p.heads;
    for(uint d=lane;d<64;d+=32){
        float angle=float(position)*pow(p.theta,-float(d)/64.0f);
        float c=cos(angle),s=sin(angle);
        float a=x[row*128+d]*inv*weight[d],b=x[row*128+d+64]*inv*weight[d+64];
        out[row*128+d]=a*c-b*s;out[row*128+d+64]=b*c+a*s;
    }
}
struct CausalArgs {uint rows,start;};
kernel void causal_attention_f32(device float *q [[buffer(0)]],device float *k [[buffer(1)]],
                                 device float *v [[buffer(2)]],device float *out [[buffer(3)]],
                                 constant CausalArgs &p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_threadgroup]]) {
    uint row=group.x,head=group.y,kv_head=head/2,end=p.start+row+1;
    float qv[4],acc[4]={0,0,0,0};
    for(uint j=0;j<4;++j)qv[j]=q[row*2048+head*128+lane+j*32];
    float maximum=-INFINITY,denom=0;
    for(uint t=0;t<end;++t){
        uint base=t*1024+kv_head*128+lane;
        float dot=0;
        for(uint j=0;j<4;++j)dot+=qv[j]*k[base+j*32];
        float score=simd_sum(dot)*rsqrt(128.0f);
        float next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        denom=denom*old+prob;maximum=next;
        for(uint j=0;j<4;++j)acc[j]=acc[j]*old+prob*v[base+j*32];
    }
    for(uint j=0;j<4;++j)out[row*2048+head*128+lane+j*32]=acc[j]/denom;
}
kernel void causal_attention_f32_h(device float *q [[buffer(0)]],device float *k [[buffer(1)]],
                                   device float *v [[buffer(2)]],device half *out [[buffer(3)]],
                                   constant CausalArgs &p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
    uint row=group.x,head=group.y,kv_head=head/2,end=p.start+row+1;
    float qv[4],acc[4]={0,0,0,0};
    for(uint j=0;j<4;++j)qv[j]=q[row*2048+head*128+lane+j*32];
    float maximum=-INFINITY,denom=0;
    for(uint t=0;t<end;++t){
        uint base=t*1024+kv_head*128+lane;
        float dot=0;
        for(uint j=0;j<4;++j)dot+=qv[j]*k[base+j*32];
        float score=simd_sum(dot)*rsqrt(128.0f);
        float next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        denom=denom*old+prob;maximum=next;
        for(uint j=0;j<4;++j)acc[j]=acc[j]*old+prob*v[base+j*32];
    }
    for(uint j=0;j<4;++j)out[row*2048+head*128+lane+j*32]=half(acc[j]/denom);
}
kernel void swiglu_f32_h(device float *gate [[buffer(0)]],device float *up [[buffer(1)]],
                         device half *out [[buffer(2)]],constant uint &count [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<count){float g=gate[i];out[i]=half((g/(1.0f+exp(-g)))*up[i]);}
}
kernel void swiglu_f32(device float *gate [[buffer(0)]],device float *up [[buffer(1)]],
                       device float *out [[buffer(2)]],constant uint &count [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<count){float g=gate[i];out[i]=(g/(1.0f+exp(-g)))*up[i];}
}
kernel void argmax_f32(device float *x [[buffer(0)]],device int *out [[buffer(1)]],
                       constant uint &width [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    float best=-INFINITY;uint index=UINT_MAX;bool invalid=false;
    for(uint j=lane;j<width;j+=32){
        float v=x[row*width+j];invalid=invalid||!isfinite(v);
        if(v>best || (v==best && j<index)){best=v;index=j;}
    }
    float maximum=simd_max(best);
    uint first=simd_min(best==maximum?index:UINT_MAX);
    bool failed=simd_any(invalid);
    if(lane==0)out[row]=failed?-1:int(first);
}
kernel void gather_rows_f32(device float *x [[buffer(0)]],device int *rows [[buffer(1)]],
                            device float *out [[buffer(2)]],constant uint2 &shape [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<shape.x*shape.y)out[i]=x[uint(rows[i/shape.y])*shape.y+i%shape.y];
}

kernel void quantize_activation_i8(device float *x [[buffer(0)]],device int8_t *q [[buffer(1)]],
                                    device float *scale [[buffer(2)]],constant uint &width [[buffer(3)]],
                                    uint tid [[thread_index_in_threadgroup]]){
    // 256 threads, cross-simdgroup maximum through threadgroup memory: max is
    // order-invariant, so widening from the original 32-lane scan is exact.
    // Each simdgroup owns a 256-wide chunk; chunks repeat every 2048 columns.
    threadgroup float stages[8];
    uint sg=tid/32,lane=tid%32,groups=min(uint(8),(width+255)/256);
    float maximum=0;
    for(uint base=sg*256;base<width;base+=2048)
        for(uint j=base+lane;j<min(base+256,width);j+=32)maximum=max(maximum,abs(x[j]));
    float partial=simd_max(maximum);
    if(groups>1){
        if(lane==0)stages[sg]=partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        partial=stages[0];
        for(uint g=1;g<groups;++g)partial=max(partial,stages[g]);
    }
    float inv=127.0f/max(partial,1e-10f);
    if(tid==0)scale[0]=partial>0?partial/127.0f:1.0f;
    for(uint base=sg*256;base<width;base+=2048)
        for(uint j=base+lane;j<min(base+256,width);j+=32)q[j]=int8_t(clamp(round(x[j]*inv),-127.0f,127.0f));
}
kernel void gemv_w8a8(device int8_t *x [[buffer(0)]],device int8_t *w [[buffer(1)]],
                       device float *out [[buffer(2)]],device float *scales [[buffer(3)]],
                       constant Shape &s [[buffer(4)]],device float *x_scale [[buffer(5)]],
                       uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
    uint row=group*8+tid/32,lane=tid%32;if(row>=s.n)return;
    int sum=0;
    for(uint k=lane*4;k<s.k;k+=128){
        int4 a=int4(*((device char4 *)(x+k))),b=int4(*((device char4 *)(w+row*s.k+k)));
        int4 product=a*b;sum+=product.x+product.y+product.z+product.w;
    }
    sum=simd_sum(sum);if(lane==0)out[row]=(float(sum)*x_scale[0])*scales[row];
}

kernel void qkv_w8a8(device int8_t *x [[buffer(0)]],device int8_t *wq [[buffer(1)]],
                      device int8_t *wk [[buffer(2)]],device int8_t *wv [[buffer(3)]],
                      device float *sq [[buffer(4)]],device float *sk [[buffer(5)]],device float *sv [[buffer(6)]],
                      device float *xs [[buffer(7)]],device float *out [[buffer(8)]],
                      uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
    uint row=group*8+tid/32,lane=tid%32;
    device int8_t *w=row<2048?wq:(row<3072?wk:wv);
    device float *scale=row<2048?sq:(row<3072?sk:sv);
    uint r=row<2048?row:(row<3072?row-2048:row-3072);int sum=0;
    for(uint k=lane*4;k<1024;k+=128){
        int4 a=int4(*((device char4 *)(x+k))),b=int4(*((device char4 *)(w+r*1024+k)));
        int4 product=a*b;sum+=product.x+product.y+product.z+product.w;
    }
    sum=simd_sum(sum);if(lane==0)out[row]=(float(sum)*xs[0])*scale[r];
}
kernel void gate_up_w8a8(device int8_t *x [[buffer(0)]],device int8_t *gate [[buffer(1)]],
                         device int8_t *up [[buffer(2)]],device float *gs [[buffer(3)]],device float *us [[buffer(4)]],
                         device float *xs [[buffer(5)]],device float *out [[buffer(6)]],
                         uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
    uint row=group*8+tid/32,lane=tid%32;int gsum=0,usum=0;
    for(uint k=lane*4;k<1024;k+=128){
        int4 a=int4(*((device char4 *)(x+k))),g=int4(*((device char4 *)(gate+row*1024+k))),u=int4(*((device char4 *)(up+row*1024+k)));
        int4 gp=a*g,uprod=a*u;gsum+=gp.x+gp.y+gp.z+gp.w;usum+=uprod.x+uprod.y+uprod.z+uprod.w;
    }
    gsum=simd_sum(gsum);usum=simd_sum(usum);
    if(lane==0){float g=(float(gsum)*xs[0])*gs[row],u=(float(usum)*xs[0])*us[row];out[row]=(g/(1.0f+exp(-g)))*u;}
}

struct ArgmaxArgs {uint width,blocks;};
kernel void argmax_blocks_f32(device float *x [[buffer(0)]],device float *maxima [[buffer(1)]],
                               device int *indices [[buffer(2)]],constant ArgmaxArgs &p [[buffer(3)]],
                               uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    uint block=group.x,row=group.y,start=block*1024,end=min(start+1024,p.width);
    float best=-INFINITY;uint index=UINT_MAX;bool invalid=false;
    for(uint j=start+lane;j<end;j+=32){
        float v=x[row*p.width+j];invalid=invalid||!isfinite(v);
        if(v>best || (v==best && j<index)){best=v;index=j;}
    }
    float maximum=simd_max(best);uint first=simd_min(best==maximum?index:UINT_MAX);bool failed=simd_any(invalid);
    if(lane==0){maxima[row*p.blocks+block]=maximum;indices[row*p.blocks+block]=failed?-1:int(first);}
}
kernel void argmax_finish_f32(device float *maxima [[buffer(0)]],device int *indices [[buffer(1)]],
                               device int *out [[buffer(2)]],constant uint &blocks [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    float best=-INFINITY;uint index=UINT_MAX;bool invalid=false;
    for(uint b=lane;b<blocks;b+=32){
        float v=maxima[row*blocks+b];int id=indices[row*blocks+b];invalid=invalid||id<0;
        if(v>best || (v==best && uint(id)<index)){best=v;index=uint(id);}
    }
    float maximum=simd_max(best);uint first=simd_min(best==maximum?index:UINT_MAX);bool failed=simd_any(invalid);
    if(lane==0)out[row]=failed?-1:int(first);
}

// ---- Fused single-token decode path ----------------------------------------
// Each kernel below reproduces the arithmetic of the unfused sequence exactly:
// integer GEMV accumulation is exact in any order, reductions use the same
// lane strides, and rounding keeps the same round()/clamp pattern.

kernel void norm_quant_i8(device float *x [[buffer(0)]],device float *weight [[buffer(1)]],
                          device int8_t *q [[buffer(2)]],device float *scale [[buffer(3)]],
                          constant float &epsilon [[buffer(4)]],
                          uint lane [[thread_index_in_threadgroup]]){
    // RMS norm of one 1024-wide row, then the shared activation quantization.
    // Same elementwise ops and simd_sum layout as norm_f32 + quantize_activation_i8;
    // y values are cached in registers instead of reloaded (bit-identical).
    float squares=0;
    for(uint j=lane;j<1024;j+=32){float v=x[j];squares+=v*v;}
    float inv=rsqrt(simd_sum(squares)/1024.0f+epsilon);
    float y[32],maximum=0;
    for(uint i=0;i<32;++i){uint j=lane+32*i;y[i]=x[j]*inv*weight[j];maximum=max(maximum,abs(y[i]));}
    maximum=simd_max(maximum);float invq=127.0f/max(maximum,1e-10f);
    if(lane==0)scale[0]=maximum>0?maximum/127.0f:1.0f;
    for(uint i=0;i<32;++i)q[lane+32*i]=int8_t(clamp(round(y[i]*invq),-127.0f,127.0f));
}

struct RopeCacheArgs {uint slot;float epsilon,theta;};
kernel void rope_cache_f32(device float *qkv [[buffer(0)]],device float *q_out [[buffer(1)]],
                           device float *k_cache [[buffer(2)]],device float *v_cache [[buffer(3)]],
                           device float *q_weight [[buffer(4)]],device float *k_weight [[buffer(5)]],
                           constant RopeCacheArgs &p [[buffer(6)]],
                           uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    // Groups 0..15: q head RMSNorm+RoPE into q_out. 16..23: k head into the KV
    // cache slot. 24..31: v copy into the cache slot. Replaces two
    // head_norm_rope dispatches and two copies; math is identical.
    if(group>=24){
        uint head=group-24;
        for(uint d=lane;d<128;d+=32)v_cache[p.slot*1024+head*128+d]=qkv[3072+head*128+d];
        return;
    }
    bool is_q=group<16;uint head=is_q?group:group-16;
    device float *src=qkv+(is_q?head*128:2048+head*128);
    device float *w=is_q?q_weight:k_weight;
    float squares=0;
    for(uint d=lane;d<128;d+=32){float v=src[d];squares+=v*v;}
    float inv=rsqrt(simd_sum(squares)/128.0f+p.epsilon);
    for(uint d=lane;d<64;d+=32){
        float angle=float(p.slot)*pow(p.theta,-float(d)/64.0f);
        float c=cos(angle),s=sin(angle);
        float a=src[d]*inv*w[d],b=src[d+64]*inv*w[d+64];
        float ra=a*c-b*s,rb=b*c+a*s;
        if(is_q){q_out[head*128+d]=ra;q_out[head*128+d+64]=rb;}
        else{k_cache[p.slot*1024+head*128+d]=ra;k_cache[p.slot*1024+head*128+d+64]=rb;}
    }
}

kernel void causal_attention_quant_i8(device float *q [[buffer(0)]],device float *k [[buffer(1)]],
                                      device float *v [[buffer(2)]],device int8_t *q8 [[buffer(3)]],
                                      device float *scale [[buffer(4)]],constant uint &start [[buffer(5)]],
                                      uint tid [[thread_index_in_threadgroup]]){
    // Single query row: 16 simdgroups, one per head, then one whole-row
    // activation quantization through threadgroup memory. The per-head loop
    // is byte-identical to causal_attention_f32 for row 0.
    threadgroup float out[2048];
    uint sg=tid/32,lane=tid%32,head=sg,kv_head=head/2,end=start+1;
    float qv[4],acc[4]={0,0,0,0};
    for(uint j=0;j<4;++j)qv[j]=q[head*128+lane+j*32];
    float maximum=-INFINITY,denom=0;
    for(uint t=0;t<end;++t){
        uint base=t*1024+kv_head*128+lane;
        float dot=0;
        for(uint j=0;j<4;++j)dot+=qv[j]*k[base+j*32];
        float score=simd_sum(dot)*rsqrt(128.0f);
        float next=max(maximum,score),old=exp(maximum-next),prob=exp(score-next);
        denom=denom*old+prob;maximum=next;
        for(uint j=0;j<4;++j)acc[j]=acc[j]*old+prob*v[base+j*32];
    }
    for(uint j=0;j<4;++j)out[head*128+lane+j*32]=acc[j]/denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top=0;
    for(uint j=lane;j<2048;j+=32)top=max(top,abs(out[j]));
    top=simd_max(top);float invq=127.0f/max(top,1e-10f);
    if(lane==0)scale[0]=top>0?top/127.0f:1.0f;
    for(uint j=lane;j<2048;j+=32)q8[j]=int8_t(clamp(round(out[j]*invq),-127.0f,127.0f));
}

kernel void gemv_w8a8_residual(device int8_t *x [[buffer(0)]],device int8_t *w [[buffer(1)]],
                               device float *residual [[buffer(2)]],device float *out [[buffer(3)]],
                               device float *scales [[buffer(4)]],constant Shape &s [[buffer(5)]],
                               device float *x_scale [[buffer(6)]],
                               uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
    // gemv_w8a8 followed by the residual add of add_f32, same operand order.
    // `precise` blocks fusing the weight-scale product with the residual add:
    // the unfused path rounds the product once in between, and an FMA here
    // shifts every layer output by 1 ulp versus the reference.
    uint row=group*8+tid/32,lane=tid%32;if(row>=s.n)return;
    int sum=0;
    for(uint k=lane*4;k<s.k;k+=128){
        int4 a=int4(*((device char4 *)(x+k))),b=int4(*((device char4 *)(w+row*s.k+k)));
        int4 product=a*b;sum+=product.x+product.y+product.z+product.w;
    }
    sum=simd_sum(sum);
    // The volatile temporary breaks the multiply-add expression tree so the
    // product rounds exactly as the unfused gemv store + add_f32 did; MSL has
    // no `precise` qualifier and contraction here shifts outputs by 1 ulp.
    volatile float prod=(float(sum)*x_scale[0])*scales[row];
    out[row]=residual[row]+prod;
}

// ---- Fused-prefill split kernels (decoder) ------------------------------------
// Decoder qkv is {rows, 4096} = q(2048)|k(1024)|v(1024); split contiguously.
kernel void split_qkv_f32(device float *src [[buffer(0)]],device float *q [[buffer(1)]],
                          device float *k [[buffer(2)]],device float *v [[buffer(3)]],
                          constant uint2 &p [[buffer(4)]],uint i [[thread_position_in_grid]]){
    // Grid covers rows*2048 (the q width); the low half of each row also
    // copies k and v so every output element is written exactly once.
    if(i>=p.x*2048)return;
    uint row=i/2048,col=i%2048;
    q[i]=src[row*4096+col];
    if(col<1024){
        k[row*1024+col]=src[row*4096+2048+col];
        v[row*1024+col]=src[row*4096+3072+col];
    }
}
// Fused gate_up GEMM output {rows, 6144}: split halves and apply SwiGLU,
// emitting half — same values swiglu_f32_h produced on separate tensors.
kernel void split_swiglu_h(device float *src [[buffer(0)]],device half *out [[buffer(1)]],
                           constant uint2 &p [[buffer(2)]],uint i [[thread_position_in_grid]]){
    if(i>=p.x*p.y)return; // p = {rows, 3072}
    uint row=i/p.y,col=i%p.y;
    float g=src[row*6144+col],u=src[row*6144+3072+col];
    out[i]=half((g/(1.0f+exp(-g)))*u);
}

// ---- INT4 MLP decode (experiment) -------------------------------------------
// Nibble layout matches quantize_w4_rows: offset-binary, value = nibble - 8,
// even element in the low nibble. Integer accumulation is exact in any order.

kernel void gemv_w4a8_residual(device int8_t *x [[buffer(0)]],device uchar *w [[buffer(1)]],
                               device float *residual [[buffer(2)]],device float *out [[buffer(3)]],
                               device float *scales [[buffer(4)]],constant Shape &s [[buffer(5)]],
                               device float *x_scale [[buffer(6)]],
                               uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
    // Grouped INT4: one scale per 64 inputs. Each lane iteration covers eight
    // elements of one group, so the group dot product is exact int arithmetic
    // and scales combine in float per iteration.
    uint row=group*8+tid/32,lane=tid%32;if(row>=s.n)return;
    device const uint *wr=(device const uint *)(w+row*(s.k/2));
    device const float *srow=scales+row*(s.k/64);
    float sum=0;
    for(uint k=lane*8;k<s.k;k+=256){
        int4 a0=int4(*((device char4 *)(x+k))),a1=int4(*((device char4 *)(x+k+4)));
        uint word=wr[k>>3];
        int4 w0=int4(int(word&15)-8,int((word>>4)&15)-8,int((word>>8)&15)-8,int((word>>12)&15)-8);
        int4 w1=int4(int((word>>16)&15)-8,int((word>>20)&15)-8,int((word>>24)&15)-8,int((word>>28)&15)-8);
        int4 p0=a0*w0,p1=a1*w1;
        sum+=float((p0.x+p0.y+p0.z+p0.w)+(p1.x+p1.y+p1.z+p1.w))*srow[k>>6];
    }
    sum=simd_sum(sum);
    volatile float prod=sum*x_scale[0];
    out[row]=residual[row]+prod;
}

kernel void gate_up_w4a8(device int8_t *x [[buffer(0)]],device uchar *gate [[buffer(1)]],
                         device uchar *up [[buffer(2)]],device float *gs [[buffer(3)]],device float *us [[buffer(4)]],
                         device float *xs [[buffer(5)]],device float *out [[buffer(6)]],
                         uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
    uint row=group*8+tid/32,lane=tid%32;
    device const uint *gr=(device const uint *)(gate+row*512),*ur=(device const uint *)(up+row*512);
    device const float *gsc=gs+row*16,*usc=us+row*16;
    float gsum=0,usum=0;
    for(uint k=lane*8;k<1024;k+=256){
        int4 a0=int4(*((device char4 *)(x+k))),a1=int4(*((device char4 *)(x+k+4)));
        uint gw=gr[k>>3],uw=ur[k>>3];
        int4 g0=int4(int(gw&15)-8,int((gw>>4)&15)-8,int((gw>>8)&15)-8,int((gw>>12)&15)-8);
        int4 g1=int4(int((gw>>16)&15)-8,int((gw>>20)&15)-8,int((gw>>24)&15)-8,int((gw>>28)&15)-8);
        int4 u0=int4(int(uw&15)-8,int((uw>>4)&15)-8,int((uw>>8)&15)-8,int((uw>>12)&15)-8);
        int4 u1=int4(int((uw>>16)&15)-8,int((uw>>20)&15)-8,int((uw>>24)&15)-8,int((uw>>28)&15)-8);
        int4 gp0=a0*g0,gp1=a1*g1,up0=a0*u0,up1=a1*u1;
        float sc_g=gsc[k>>6],sc_u=usc[k>>6];
        gsum+=float((gp0.x+gp0.y+gp0.z+gp0.w)+(gp1.x+gp1.y+gp1.z+gp1.w))*sc_g;
        usum+=float((up0.x+up0.y+up0.z+up0.w)+(up1.x+up1.y+up1.z+up1.w))*sc_u;
    }
    gsum=simd_sum(gsum);usum=simd_sum(usum);
    if(lane==0){float g=gsum*xs[0],u=usum*xs[0];out[row]=(g/(1.0f+exp(-g)))*u;}
}

// Audio frontend: periodic Hann, center reflection, 400-point DFT, Slaney Mel.
// Reference uses a 400-point DFT, so this correctness-first path keeps the same
// transform (including FP32 table angles) before considering an FFT replacement.
inline float mel_hz(float m){
    return m>=15.0f?1000.0f*exp((log(6.4f)/27.0f)*(m-15.0f)):200.0f*m/3.0f;
}
kernel void frontend_tables(device float *cosines [[buffer(0)]],device float *sines [[buffer(1)]],
                             device float *window [[buffer(2)]],device float *filters [[buffer(3)]],
                             uint i [[thread_position_in_grid]]){
    const float pi=3.14159265358979323846f;
    if(i<201*400){
        float angle=((2.0f*pi)*float(i/400))*float(i%400)/400.0f;
        cosines[i]=cos(angle);sines[i]=sin(angle);
    }
    if(i<400)window[i]=0.5f*(1.0f-cos((2.0f*pi)*float(i)/400.0f));
    if(i<128*201){
        uint m=i/201,k=i%201;float maximum=15.0f+log(8.0f)*(27.0f/log(6.4f));
        float left=mel_hz(maximum*float(m)/129.0f),center=mel_hz(maximum*float(m+1)/129.0f),right=mel_hz(maximum*float(m+2)/129.0f);
        float frequency=float(k)*8000.0f/200.0f;
        filters[i]=max(0.0f,min((frequency-left)/(center-left),(right-frequency)/(right-center)))*(2.0f/(right-left));
    }
}
struct FrontendArgs {uint samples,frames;};
kernel void dft_power_f32(device float *pcm [[buffer(0)]],device float *cosines [[buffer(1)]],
                          device float *sines [[buffer(2)]],device float *window [[buffer(3)]],
                          device float *power [[buffer(4)]],constant FrontendArgs &p [[buffer(5)]],
                          uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    uint frame=group.x,k=group.y;float real=0,imag=0;
    for(uint n=lane;n<400;n+=32){
        int sample=int(frame*160+n)-200;
        if(sample<0)sample=-sample;
        else if(sample>=int(p.samples))sample=2*int(p.samples)-2-sample;
        float value=(sample>=0 && sample<int(p.samples))?pcm[sample]*window[n]:0.0f;
        real+=value*cosines[k*400+n];imag+=value*sines[k*400+n];
    }
    real=simd_sum(real);imag=simd_sum(imag);
    if(lane==0)power[frame*201+k]=real*real+imag*imag;
}
kernel void mel_log_f32(device float *power [[buffer(0)]],device float *filters [[buffer(1)]],
                        device float *mel [[buffer(2)]],constant uint &frames [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    uint frame=group.x,m=group.y;float sum=0;
    for(uint k=lane;k<201;k+=32)sum+=power[frame*201+k]*filters[m*201+k];
    sum=simd_sum(sum);
    if(lane==0)mel[m*frames+frame]=log10(max(sum,1e-10f));
}
kernel void mel_max_f32(device float *mel [[buffer(0)]],device float *maximum [[buffer(1)]],
                        constant uint &count [[buffer(2)]],uint lane [[thread_index_in_threadgroup]]){
    float best=-INFINITY;for(uint i=lane;i<count;i+=32)best=max(best,mel[i]);
    best=simd_max(best);if(lane==0)maximum[0]=best;
}
// Two-level max for the full spectrogram: the original single-32-lane group
// serialized hundreds of thousands of strided loads; max is order-invariant,
// so partials commute exactly.
kernel void mel_max_blocks_f32(device float *mel [[buffer(0)]],device float *partials [[buffer(1)]],
                               constant uint2 &p [[buffer(2)]],uint group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]){
    constexpr uint span=8192;threadgroup float tile[256];
    uint start=group*span,end=min(start+span,p.x);
    float best=-INFINITY;
    for(uint i=start+tid;i<end;i+=256)best=max(best,mel[i]);
    tile[tid]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<32){
        float v=-INFINITY;
        for(uint j=tid;j<256;j+=32)v=max(v,tile[j]);
        v=simd_max(v); // whole simdgroup must participate in the reduction
        if(tid==0)partials[group]=v;
    }
}
kernel void mel_max_finish_f32(device float *partials [[buffer(0)]],device float *maximum [[buffer(1)]],
                               constant uint &blocks [[buffer(2)]],uint lane [[thread_index_in_threadgroup]]){
    float best=-INFINITY;for(uint b=lane;b<blocks;b+=32)best=max(best,partials[b]);
    best=simd_max(best);if(lane==0)maximum[0]=best;
}
kernel void mel_normalize_f32(device float *mel [[buffer(0)]],device float *maximum [[buffer(1)]],
                              constant uint &count [[buffer(2)]],uint i [[thread_position_in_grid]]){
    if(i<count)mel[i]=(max(mel[i],maximum[0]-8.0f)+4.0f)*0.25f;
}

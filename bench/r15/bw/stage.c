// Emulate one decode token's memory traffic: 28 layers x {QKV 4MB, [attn KV 3.5MB on <=8 thr], O 2MB, gate_up 6MB, down 3MB}
// + lm_head 155.6MB. mode 0: flat stream of the same bytes (1 barrier). mode 1: staged with barrier per stage
// + EXTRA extra barriers per layer, each preceded by GLUE_NS of tid-0-only work (emulates serial glue stages).
#include <arm_neon.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
#define MB (1UL<<20)
static int NT, EXTRA, MODE, GLUE_NS;
static _Atomic int arrived=0; static _Atomic int gen=0;
static inline void barrier(void){ int g=atomic_load_explicit(&gen,memory_order_acquire); if(atomic_fetch_add_explicit(&arrived,1,memory_order_acq_rel)==NT-1){atomic_store_explicit(&arrived,0,memory_order_relaxed);atomic_fetch_add_explicit(&gen,1,memory_order_release);} else while(atomic_load_explicit(&gen,memory_order_acquire)==g){} }
static int8_t* L[28][5]; static size_t SZ[5]={4*MB,3584*1024,2*MB,6*MB,3*MB}; static int8_t* LM; static size_t LMSZ=155*MB+614*1024;
static int8_t* FLAT; static size_t FLATSZ;
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static inline void spin_ns(int ns){ double t0=now(); while((now()-t0)*1e9<ns){} }
static __attribute__((noinline)) int32x4_t rd(int32x4_t acc, const int8_t* s, size_t n){ __asm__ volatile("" ::: "memory"); int8x16_t w=vdupq_n_s8(1); int32x4_t a1=vdupq_n_s32(0);
  for(size_t i=0;i+32<=n;i+=32){ acc=vdotq_s32(acc,vld1q_s8(s+i),w); a1=vdotq_s32(a1,vld1q_s8(s+i+16),w);} return vaddq_s32(acc,a1); }
static double times[64]; static int32_t sinks[64];
static void* worker(void* p){ int t=(int)(intptr_t)p; int32x4_t acc=vdupq_n_s32(0); double best=1e9,sum=0; int n=0;
  for(int rep=0;rep<14;rep++){ barrier(); double t0=now();
    if(MODE==0){ size_t c=(FLATSZ/NT)&~63UL; acc=rd(acc,FLAT+t*c,c); }
    else for(int l=0;l<28;l++){ for(int s=0;s<5;s++){ int parts = (s==1)? (NT<8?NT:8) : NT; if(t<parts){ size_t c=(SZ[s]/parts)&~63UL; acc=rd(acc,L[l][s]+t*c,c);} barrier(); }
      for(int e=0;e<EXTRA;e++){ if(t==0 && GLUE_NS) spin_ns(GLUE_NS); barrier(); } }
    if(MODE==1){ size_t c=(LMSZ/NT)&~63UL; acc=rd(acc,LM+t*c,c); }
    barrier(); double dt=now()-t0; if(rep>1){ if(dt<best)best=dt; sum+=dt; n++; } }
  if(t==0){ times[0]=best; times[1]=sum/n; } sinks[t]=vaddvq_s32(acc); return 0; }
int main(int argc,char**argv){ assert(argc >= 3); NT=atoi(argv[1]); MODE=atoi(argv[2]); EXTRA=argc>3?atoi(argv[3]):0; GLUE_NS=argc>4?atoi(argv[4]):0;
  assert(NT >= 1 && NT <= 64 && (MODE == 0 || MODE == 1) && EXTRA >= 0 && GLUE_NS >= 0);
  double bytes=0; for(int s=0;s<5;s++) bytes+=28.0*SZ[s]; bytes+=LMSZ;
  if(MODE==0){ FLATSZ=(size_t)bytes; posix_memalign((void**)&FLAT,1<<14,FLATSZ); memset(FLAT,1,FLATSZ); }
  else { for(int l=0;l<28;l++) for(int s=0;s<5;s++){ posix_memalign((void**)&L[l][s],1<<14,SZ[s]); memset(L[l][s],1,SZ[s]); }
         posix_memalign((void**)&LM,1<<14,LMSZ); memset(LM,1,LMSZ); }
  pthread_t th[64]; for(int t=0;t<NT;t++) pthread_create(&th[t],0,worker,(void*)(intptr_t)t);
  for(int t=0;t<NT;t++) pthread_join(th[t],0);
  long long ss=0; for(int t=0;t<NT;t++) ss+=sinks[t]; printf("[sink %lld] thr=%2d mode=%s extra/layer=%d glue=%dns  best %.2f ms (%.0f GB/s)  mean %.2f ms\n",ss,NT,MODE?"staged":"flat  ",EXTRA,GLUE_NS,times[0]*1e3,bytes/times[0]/1e9,times[1]*1e3); return 0; }

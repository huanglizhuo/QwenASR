// Per-token decode emulation (weights only, 575MB): 28 layers x stages {QKV 4MB, O 2MB, gate_up 6MB, down 3MB} + lm_head.
// Each weight stage is followed by a "glue" stage of GLUE ns (+barrier), like the real fused region.
// VAR 1: glue on tid0 only, others wait at barrier (current design)
// VAR 2: glue replicated on all threads, no glue barrier
// VAR 3: like 1, but each thread prefetches PF KB of its NEXT stage slice before waiting/gluing
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
static int NT, VAR, GLUE, PF;
static _Atomic int arrived=0; static _Atomic int gen=0;
static inline void barrier(void){ int g=atomic_load_explicit(&gen,memory_order_acquire); if(atomic_fetch_add_explicit(&arrived,1,memory_order_acq_rel)==NT-1){atomic_store_explicit(&arrived,0,memory_order_relaxed);atomic_fetch_add_explicit(&gen,1,memory_order_release);} else while(atomic_load_explicit(&gen,memory_order_acquire)==g){} }
#define NS 4
static int8_t* L[28][NS]; static size_t SZ[NS]={4*MB,2*MB,6*MB,3*MB}; static int8_t* LM; static size_t LMSZ=155*MB+614*1024;
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static inline void spin_ns(int ns){ if(ns<=0) return; double t0=now(); while((now()-t0)*1e9<ns){} }
static __attribute__((noinline)) int32x4_t rd(int32x4_t acc, const int8_t* s, size_t n){ __asm__ volatile("" ::: "memory"); int8x16_t w=vdupq_n_s8(1); int32x4_t a1=vdupq_n_s32(0);
  for(size_t i=0;i+32<=n;i+=32){ acc=vdotq_s32(acc,vld1q_s8(s+i),w); a1=vdotq_s32(a1,vld1q_s8(s+i+16),w);} return vaddq_s32(acc,a1); }
static inline void pf(const int8_t* s, size_t n){ for(size_t i=0;i<n;i+=128) __builtin_prefetch(s+i,0,0); }
static double times[2]; static int32_t sinks[64];
static void* worker(void* p){ int t=(int)(intptr_t)p; int32x4_t acc=vdupq_n_s32(0); double best=1e9,sum=0; int n=0;
  for(int rep=0;rep<14;rep++){ barrier(); double t0=now();
    for(int l=0;l<28;l++) for(int s=0;s<NS;s++){
      size_t c=(SZ[s]/NT)&~63UL; acc=rd(acc,L[l][s]+t*c,c); barrier();
      // next stage slice
      const int8_t* nx; size_t nc;
      if(s+1<NS){ nc=(SZ[s+1]/NT)&~63UL; nx=L[l][s+1]+t*nc; } else if(l+1<28){ nc=(SZ[0]/NT)&~63UL; nx=L[l+1][0]+t*nc; } else { nc=(LMSZ/NT)&~63UL; nx=LM+t*nc; }
      if(VAR==3 && PF>0) pf(nx, (size_t)PF*1024 < nc ? (size_t)PF*1024 : nc);
      if(VAR==1||VAR==3){ if(t==0) spin_ns(GLUE); barrier(); }
      else if(VAR==2){ spin_ns(GLUE); }
    }
    size_t c=(LMSZ/NT)&~63UL; acc=rd(acc,LM+t*c,c);
    barrier(); double dt=now()-t0; if(rep>1){ if(dt<best)best=dt; sum+=dt; n++; } }
  if(t==0){ times[0]=best; times[1]=sum/n; } sinks[t]=vaddvq_s32(acc); return 0; }
int main(int argc,char**argv){ assert(argc >= 4); NT=atoi(argv[1]); VAR=atoi(argv[2]); GLUE=atoi(argv[3]); PF=argc>4?atoi(argv[4]):0;
  assert(NT >= 1 && NT <= 64 && VAR >= 1 && VAR <= 3 && GLUE >= 0 && PF >= 0);
  double bytes=LMSZ; for(int s=0;s<NS;s++) bytes+=28.0*SZ[s];
  for(int l=0;l<28;l++) for(int s=0;s<NS;s++){ posix_memalign((void**)&L[l][s],1<<14,SZ[s]); memset(L[l][s],1,SZ[s]); }
  posix_memalign((void**)&LM,1<<14,LMSZ); memset(LM,1,LMSZ);
  pthread_t th[64]; for(int t=0;t<NT;t++) pthread_create(&th[t],0,worker,(void*)(intptr_t)t);
  for(int t=0;t<NT;t++) pthread_join(th[t],0);
  long long ss=0; for(int t=0;t<NT;t++) ss+=sinks[t];
  printf("[%lld] thr=%2d var=%d glue=%4dns pf=%3dKB  best %.2f ms (%.0f GB/s)  mean %.2f ms\n",ss&1,NT,VAR,GLUE,PF,times[0]*1e3,bytes/times[0]/1e9,times[1]*1e3); return 0; }

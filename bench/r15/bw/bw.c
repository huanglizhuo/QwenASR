#include <arm_neon.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
#define TOTAL (576UL<<20)
static int8_t *buf; static int NT;
static _Atomic int arrived=0; static _Atomic int gen=0;
static void barrier(int n){ int g=atomic_load(&gen); if(atomic_fetch_add(&arrived,1)==n-1){atomic_store(&arrived,0);atomic_fetch_add(&gen,1);} else while(atomic_load(&gen)==g){} }
typedef struct { int tid; int32_t sink; } arg_t;
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void* worker(void* p){ arg_t* a=p; size_t chunk=(TOTAL/NT)&~63UL; int8_t* s=buf+(size_t)a->tid*chunk;
  int8x16_t w = vdupq_n_s8(1); int32x4_t acc0=vdupq_n_s32(0),acc1=acc0,acc2=acc0,acc3=acc0;
  for(int rep=0;rep<8;rep++){ barrier(NT+1);
    for(size_t i=0;i<chunk;i+=64){ acc0=vdotq_s32(acc0,vld1q_s8(s+i),w); acc1=vdotq_s32(acc1,vld1q_s8(s+i+16),w);
      acc2=vdotq_s32(acc2,vld1q_s8(s+i+32),w); acc3=vdotq_s32(acc3,vld1q_s8(s+i+48),w);} barrier(NT+1);}
  a->sink=vaddvq_s32(vaddq_s32(vaddq_s32(acc0,acc1),vaddq_s32(acc2,acc3))); return 0; }
int main(int argc,char**argv){ assert(argc >= 2); NT=atoi(argv[1]); assert(NT >= 1 && NT <= 64); posix_memalign((void**)&buf,1<<14,TOTAL); memset(buf,1,TOTAL);
  pthread_t th[64]; arg_t a[64];
  for(int t=0;t<NT;t++){a[t].tid=t; pthread_create(&th[t],0,worker,&a[t]);}
  double best=1e9, sum=0; for(int rep=0;rep<8;rep++){ double t0=now(); barrier(NT+1); barrier(NT+1); double dt=now()-t0; if(rep>1){ if(dt<best)best=dt; sum+=dt;} }
  for(int t=0;t<NT;t++) pthread_join(th[t],0);
  printf("threads=%2d  best %.1f GB/s  mean %.1f GB/s  (best %.2f ms per 576MB)\n",NT,((TOTAL/NT)&~63UL)*NT/best/1e9,((TOTAL/NT)&~63UL)*NT/(sum/6)/1e9,best*1e3); return 0; }

#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { double gpu_ms; uint32_t audio_tokens, decoder_positions, generated; } QMetalStats;
void *qmetal_create(const char *package,const char *kernels,char *error,size_t error_capacity);
void qmetal_destroy(void *handle);
// Tokens/text are host control data. PCM, Mel, encoder, embedding, decoder,
// KV, logits and reductions run entirely on Metal; only output IDs return.
int qmetal_infer(void *handle,const float *pcm,size_t samples,
                 const int32_t *prefix,size_t prefix_count,const int32_t *suffix,size_t suffix_count,
                 const int32_t *timestamp_rows,size_t row_count,uint32_t max_new_tokens,
                 int32_t *output,size_t output_capacity,QMetalStats *stats,char *error,size_t error_capacity);
#ifdef __cplusplus
}
#endif

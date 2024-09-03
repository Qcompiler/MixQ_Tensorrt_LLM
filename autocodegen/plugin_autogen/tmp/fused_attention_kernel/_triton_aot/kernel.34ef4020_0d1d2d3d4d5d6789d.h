#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d(void);
void load_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d(void);
// tt-linker: fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d:CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len:128x64x128_warps4xstages2
CUresult fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len);

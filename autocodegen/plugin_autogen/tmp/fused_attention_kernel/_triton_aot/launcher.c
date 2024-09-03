#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: fused_attention_kernel_128x64x128_warps4xstages2
CUresult fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len);
CUresult fused_attention_kernel_fbf0f274_0d1d2d3d4d5d6789(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len);

CUresult fused_attention_kernel_128x64x128_warps4xstages2(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len){
  if ((Out % 16 == 0) && (L % 16 == 0) && (M % 16 == 0) && (Q % 16 == 0) && (K % 16 == 0) && (V % 16 == 0) && (seq_len % 16 == 0))
    return fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d(stream, Out, L, M, Q, K, V, sm_scale, batch_size, num_heads, seq_len);
  if ((Out % 16 == 0) && (L % 16 == 0) && (M % 16 == 0) && (Q % 16 == 0) && (K % 16 == 0) && (V % 16 == 0))
    return fused_attention_kernel_fbf0f274_0d1d2d3d4d5d6789(stream, Out, L, M, Q, K, V, sm_scale, batch_size, num_heads, seq_len);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: fused_attention_kernel_128x64x128_warps4xstages2
void load_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d();
void load_fused_attention_kernel_fbf0f274_0d1d2d3d4d5d6789();
void load_fused_attention_kernel_128x64x128_warps4xstages2() {
  load_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d();
  load_fused_attention_kernel_fbf0f274_0d1d2d3d4d5d6789();
}

// unload for: fused_attention_kernel_128x64x128_warps4xstages2
void unload_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d();
void unload_fused_attention_kernel_fbf0f274_0d1d2d3d4d5d6789();
void unload_fused_attention_kernel_128x64x128_warps4xstages2() {
  unload_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d();
  unload_fused_attention_kernel_fbf0f274_0d1d2d3d4d5d6789();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len);
kernel_func_t fused_attention_kernel_kernels[] = {
  fused_attention_kernel_128x64x128_warps4xstages2,
};

int fused_attention_kernel_get_num_algos(void){
  return (int)sizeof(fused_attention_kernel_kernels);
}

CUresult fused_attention_kernel(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len, int algo_id){
  assert (algo_id < (int)sizeof(fused_attention_kernel_kernels));
  return fused_attention_kernel_kernels[algo_id](stream, Out, L, M, Q, K, V, sm_scale, batch_size, num_heads, seq_len);
}

void load_fused_attention_kernel(void){
  load_fused_attention_kernel_128x64x128_warps4xstages2();
}

void unload_fused_attention_kernel(void){
  unload_fused_attention_kernel_128x64x128_warps4xstages2();
}


CUresult fused_attention_kernel_default(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len){
  return fused_attention_kernel(stream, Out, L, M, Q, K, V, sm_scale, batch_size, num_heads, seq_len, 0);
}

#include <cuda.h>

CUresult fused_attention_kernel_128x64x128_warps4xstages2(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len);
void load_fused_attention_kernel_128x64x128_warps4xstages2();
void unload_fused_attention_kernel_128x64x128_warps4xstages2();
    
int fused_attention_kernel_get_num_algos(void);

CUresult fused_attention_kernel_default(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len);
CUresult fused_attention_kernel(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len, int algo_id);
void load_fused_attention_kernel();
void unload_fused_attention_kernel();
    
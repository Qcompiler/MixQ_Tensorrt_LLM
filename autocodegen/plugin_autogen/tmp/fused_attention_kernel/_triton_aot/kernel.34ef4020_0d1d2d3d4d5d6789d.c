/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <cuda.h>


// helpers to check for cuda errors
#define CUDA_CHECK(ans) {\
    gpuAssert((ans), __FILE__, __LINE__);\
  }\

static inline void gpuAssert(CUresult code, const char *file, int line) {
  if (code != CUDA_SUCCESS) {
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);
    char err[1024] = {0};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }
}

// globals
#define CUBIN_NAME fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_cubin
CUmodule fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_mod = NULL;
CUfunction fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_func = NULL;


void unload_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d(void) {
    CUDA_CHECK(cuModuleUnload(fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_mod));
}

// TODO: some code duplication with `runtime/backend/cuda.c`
void load_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d() {
    int dev = 0;
    void *bin = (void *)&CUBIN_NAME;
    int shared = 49154;
    CUDA_CHECK(cuModuleLoadData(&fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_mod, bin));
    CUDA_CHECK(cuModuleGetFunction(&fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_func, fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_mod, "fused_attention_kernel_0d1d2d3d4d5d6789d"));
    // set dynamic shared memory if necessary
    int shared_optin;
    CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    if (shared > 49152 && shared_optin > 49152) {
      CUDA_CHECK(cuFuncSetCacheConfig(fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_func, CU_FUNC_CACHE_PREFER_SHARED));
      CUDA_CHECK(cuFuncSetAttribute(fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
    }
}

/*
['BLOCK_M=128', 'BLOCK_DMODEL=64', 'BLOCK_N=128', 'num_warps=4', 'num_stages=2']
*/
CUresult fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d(CUstream stream, CUdeviceptr Out, CUdeviceptr L, CUdeviceptr M, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, float sm_scale, int32_t batch_size, int32_t num_heads, int32_t seq_len) {
    if (fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_func == NULL)
       load_fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d();
    unsigned int gX = (seq_len + 127) / 128;
    unsigned int gY = batch_size * num_heads;
    unsigned int gZ = 1;
    void *args[10] = { &Out, &L, &M, &Q, &K, &V, &sm_scale, &batch_size, &num_heads, &seq_len };
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return cuLaunchKernel(fused_attention_kernel_34ef4020_0d1d2d3d4d5d6789d_func, gX, gY, gZ, 4 * 32, 1, 1, 49154, stream, args, NULL);
}
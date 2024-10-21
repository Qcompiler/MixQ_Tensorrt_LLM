## 1. 文件/models/modeling_utils.py 的修改内容为：
``` 101,102d100
<             QuantAlgo.int8_mix: 'int8_mix',
<             QuantAlgo.int4_mix: 'int4_mix',
387d384
<         print("call __post_init__")
433,436d429
< 
<         print("load weight")
<         print("lianxiang/lianxiangTRT/tensorrt_llm/models/modeling_utils.py")
<  
454,462d446
< 
<         #print("tensorrt_llm/models/modeling_utils.py provided_names")
<         #print(provided_names)
< 
<         #print(weights)
< 
<         #print("required names")
<         #print(required_names)
<  
665d648
< 
``` 
## 2. 文件/models/llama/model.py 的修改内容为：
``` 379,381c379
<             QuantAlgo.W4A8_AWQ,
<             QuantAlgo.int8_mix,
<             QuantAlgo.int4_mix,
---
>             QuantAlgo.W4A8_AWQ
388,389d385
<         print("check")
<         print(quant_config.quant_algo in DEFAULT_MODELOPT_FLOW)
``` 
## 3. 文件/models/qwen/model.py 的修改内容为：
``` 348,350c348
<             QuantAlgo.W4A8_AWQ,
<             QuantAlgo.int8_mix,
<             QuantAlgo.int4_mix,
---
>             QuantAlgo.W4A8_AWQ
``` 
## 4. 文件/quantization/quantize_by_modelopt.py 的修改内容为：
``` 97d96
< 
108d106
<         "int8_mix" : EMPTY_CFG
396,397d393
<     print("tensorrt_llm/quantization/quantize_by_modelopt.py")
<     print("quantize_and_export")
414,421d409
<         if "mix" not in qformat:
<             calib_dataloader = get_calib_dataloader(
<                 dataset_name_or_dir=calib_dataset,
<                 tokenizer=tokenizer,
<                 batch_size=batch_size,
<                 calib_size=calib_size,
<                 block_size=calib_max_seq_length,
<             )
423,424c411,418
<         else:
<             calib_dataloader = {}
---
>         calib_dataloader = get_calib_dataloader(
>             dataset_name_or_dir=calib_dataset,
>             tokenizer=tokenizer,
>             batch_size=batch_size,
>             calib_size=calib_size,
>             block_size=calib_max_seq_length,
>         )
> 
456d449
<         print("----------export_tensorrt_llm_checkpoint--------------")
``` 
## 5. 文件/quantization/quantize.py 的修改内容为：
``` 269d268
< 
276d274
< 
278,286c276
<         if quant_mode.has_mix_quant():
<             model = mix_quantize(model, quant_config)    
<         else:
<             model = smooth_quantize(model, quant_config)
<     # elif quant_mode.is_weight_only():
<     #     if quant_mode.has_per_group_scaling():
<     #         model = weight_only_groupwise_quantize(model, quant_config)
<     #     else:
<     #         model = weight_only_quantize(model, quant_config)
---
>         model = smooth_quantize(model, quant_config)
288,290c278,281
<         print("weight only mixq")
<         model = mix_quantize(model, quant_config)  
< 
---
>         if quant_mode.has_per_group_scaling():
>             model = weight_only_groupwise_quantize(model, quant_config)
>         else:
>             model = weight_only_quantize(model, quant_config)
296,350d286
< 
< 
< def mix_quantize(model, quant_config: QuantConfig):
< 
<     print("----mix_quantize----------")
<  
< 
<     return mix_quantize_ootb(model, quant_config)
< 
< from plugin import  MixQLinear
< 
< def mix_quantize_ootb(
<     model,
<     quant_config: QuantConfig,
<     current_key_name=None,
< ):
<     exclude_modules = quant_config.exclude_modules or ['lm_head']
<     
<     # print(model)
<     # exit()
<     for name, module in model.named_children():
<         if current_key_name is None:
<             current_key_name = []
<         current_key_name.append(name)
<         
< 
<         if len(list(module.children())) > 0:
< 
<             mix_quantize_ootb(module, quant_config, current_key_name)
< 
<         
<         print(name)
<         if "qkv" in name or "gate" in name or "proj" in name:
<             print(" use mix quant to quant the qkv projection!")
<             
<             if isinstance(module, ColumnLinear) and name not in exclude_modules:
<                 if not any(key in '.'.join(current_key_name)
<                         for key in exclude_modules):
<                     model._modules[name] = MixQLinear(
<                         module.in_features, module.out_features * module.tp_size,
<                         module.bias, module.dtype, module.tp_group, module.tp_size,
<                         module.gather_output)
<                     
<             elif isinstance(module, RowLinear) and name not in exclude_modules:
<                 if not any(key in '.'.join(current_key_name)
<                         for key in exclude_modules):
<                     assert module.tp_size == 1
<                     model._modules[name] = MixQLinear(
<                         module.in_features * module.tp_size, module.out_features,
<                         module.bias, module.dtype, module.tp_group, module.tp_size)
< 
<         current_key_name.pop(-1)
< 
<     setattr(model, 'quant_mode', quant_config.quant_mode)
<     return model
\ No newline at end of file
``` 
## 6. 文件/quantization/mode.py 的修改内容为：
``` 37,38d36
<     int8_mix = auto()
<     int4_mix = auto()
75,76d72
<     MIX_PRECISION = auto()
<     
109,112d104
<     def has_mix_quant(self):
<         return self._any(self.MIX_PRECISION)
<     
< 
149c141
<                          | self.FP8_QDQ | self.MIX_PRECISION)
---
>                          | self.FP8_QDQ | self.FP8_ROWWISE)
238,245d229
<     def use_mix_precision(use_int4_weights=False):
<         print("------use_mix_precision---------")
<         tmp =  QuantMode.from_description(True, False, False, False)
<         tmp = tmp | QuantMode.MIX_PRECISION
< 
<         return tmp
<     
<     @staticmethod
259,262d242
<         print("tensorrt_llm/quantization/mode.py")
<         print("quant_algo is ")
< 
<         print(quant_algo)
267,277d246
<         elif quant_algo == QuantAlgo.int8_mix:
<             print("quant_algo is int8 mix")
<             print("tensorrt_llm/quantization/mode.py")
<  
<             quant_mode = QuantMode.use_mix_precision(use_int4_weights=False)
<         elif quant_algo == QuantAlgo.int4_mix:
<             print("quant_algo is int4 mix")
<             print("tensorrt_llm/quantization/mode.py")
<             quant_mode = QuantMode.use_mix_precision(use_int4_weights=True)     
< 
< 
279d247
< 
``` 
## 7. 文件/runtime/model_runner_cpp.py 的修改内容为：
``` 226,228c226
<         print("------max-batch is ------")
<         print(max_batch_size)
<         print(model_config.max_batch_size)
---
> 
``` 


## 8. 文件/torch/export/layer_utils.py 的修改内容为：
``` 43d42
<     QUANTIZATION_INT8_MIX,
1507,1513c1506,1509
< 
<                 if len(w_quantizer.block_sizes) > 0:
<                     assert (
<                         len(w_quantizer.block_sizes) > 0 and w_quantizer.block_sizes[-1] > 0
<                     ), "Invalid block_sizes"
<                     return QUANTIZATION_INT4_AWQ
<                 return QUANTIZATION_INT4_MIX
---
>                 assert (
>                     len(w_quantizer.block_sizes) > 0 and w_quantizer.block_sizes[-1] > 0
>                 ), "Invalid block_sizes"
>                 return QUANTIZATION_INT4_AWQ
1515,1517c1511
<                 return QUANTIZATION_INT8_MIX
<             elif w_quantizer.num_bits == (8, 16):
<                 return QUANTIZATION_INT8_MIX
---
>                 return QUANTIZATION_INT8_SQ
``` 
## 9. 文件/torch/export/model_config_utils.py 的修改内容为：
``` 26d25
<     QUANTIZATION_INT8_MIX,
306,308d304
<     assert quantization == QUANTIZATION_INT8_MIX
<     if quantization == QUANTIZATION_INT8_MIX:
<         return (weight / weights_scaling_factor[:, None]).round().clamp(-128, 127).to(torch.int8)
379a376,386
> 
>     def _linear_layer_to_quantized_weight(linear_layers):
>         for linear_layer in linear_layers:
>             if isinstance(linear_layer, LinearConfig):
>                 if linear_layer.weights_scaling_factor is not None:
>                     linear_layer.weight = to_quantized_weight(
>                         linear_layer.weight,
>                         linear_layer.weights_scaling_factor,
>                         model_config.quantization,
>                     )
> 
382,403d388
<     print("pack_linear_weights")
<      
<     #print(model_config)
< 
< 
<     #relative_path = "act_scales/%s.pt"%(model_config._name_or_path.split("/")[-1])
< 
<     #relative_path = "act_scales/Llama-2-7b.pt"
<     #relative_path = "act_scales/qwen2-7b-instruct.pt"
<     relative_path = "act_scales/Qwen2-72B.pt"
<     print("-----load----relative_path-",relative_path)
<     act_scales = torch.load(relative_path)
<     # from safetensors import safe_open
<     # awq_model = torch.load("/code/tensorrt_llm/manual_plugin/checkpoint/Llama-2-7b-w4-g128-v2.pt",
<     #                       map_location=torch.device('cpu'))
<         
<     names = [  "self_attn.q_proj",
<                 "mlp.gate_proj",
<                 "mlp.up_proj",
<                 ]
<     layer_id = -1
<     #print(len(model_config.layers))
405,406c390
<     torch.cuda.empty_cache()
<     layer_id = -1
---
>     attention_key_list = ["attention", "self_attention", "cross_attention"]
407a392,419
>         linear_layers = []
>         if any([hasattr(decoder_config, attention_key) for attention_key in attention_key_list]):
>             for attention_key in attention_key_list:
>                 attention = getattr(decoder_config, attention_key, None)
>                 if attention:
>                     linear_layers = [
>                         attention.qkv,
>                         attention.dense,
>                     ]
>         if decoder_config.recurrent:
>             linear_layers = [
>                 decoder_config.recurrent.linear_y,
>                 decoder_config.recurrent.linear_x,
>                 decoder_config.recurrent.linear_out,
>             ]
> 
>         if isinstance(decoder_config.mlp, MOEConfig):
>             if model_config.quantization not in [QUANTIZATION_FP8, QUANTIZATION_INT4_AWQ]:
>                 raise NotImplementedError(
>                     f"MOE quantization for {model_config.quantization} is not supported yet."
>                 )
>             else:
>                 linear_layers.append(decoder_config.mlp.experts.fc)
>                 linear_layers.append(decoder_config.mlp.experts.proj)
>         else:
>             linear_layers.append(decoder_config.mlp.fc)
>             linear_layers.append(decoder_config.mlp.proj)
>             linear_layers.append(decoder_config.mlp.gate)
409,430c421
<         linear_layers = [
<                         decoder_config.attention.qkv ,
<                         # decoder_config.attention.dense,
<                         # decoder_config.mlp.fc,
<                         decoder_config.mlp.gate,
<                         decoder_config.mlp.proj,
<                     ]  
<         layer_id += 1
<         print("layer id is ", layer_id)
<         for i, linear_layer in enumerate(linear_layers): 
< 
<             if isinstance(linear_layer, LinearConfig):
<                 
<                 
<                 name = names[i]
<                 
<                 layer_scales = act_scales['model.layers.{}.{}'.format(layer_id, name)]
< 
<                 print("layer_scales")
<                  
<                 linear_layer.weights_scaling_factor =   (torch.max(torch.abs(linear_layer.weight), dim=1)[0].unsqueeze(1) / (
<                         127)).to(torch.float16).reshape((linear_layer.weight.shape[0],))
---
>         _linear_layer_to_quantized_weight(linear_layers)
432,434c423,424
<                 if linear_layer.weights_scaling_factor is not None:
<                     import mixlib
<                     from EETQ import quant_weights, preprocess_weights, w8_a16_gemm
---
>     if model_config.medusa_heads is not None:
>         linear_layers = []
436,529c426,429
<                     # 一个int 放在2个half 中
<                     int8_weight_cpu = torch.t(linear_layer.weight.data).contiguous().cpu()
<                     int8_weight, scales = quant_weights(int8_weight_cpu, torch.int8, False)
<                     
<                     linear_layer.qweight = mixlib.int8_matrix_to_half(int8_weight.cuda()).cpu()
<                     linear_layer.scales = scales
<                     print("-----qweight-----")
<                     print(linear_layer.qweight.shape)
<                     print(linear_layer.scales.shape)
< 
<                     fp_features = 128
< 
<                     linear_layer.fp_ind =  torch.sort(layer_scales)[1][-fp_features:] 
< 
<                     #print([layer_scales[linear_layer.fp_ind]])
<                  
<                     linear_layer.fp_weight = linear_layer.weight[:, linear_layer.fp_ind]
<                     linear_layer.weight[:, linear_layer.fp_ind] *= 0 # setting to zeros
<                     # 转成short
<                     import mixlib
<                     # 一个int 放在2个half 中
<                     linear_layer.fp_ind = mixlib.int_to_half(linear_layer.
<                     fp_ind.to(torch.int32).cuda()).cpu()
<                     
<                     linear_layer.weight = to_quantized_weight(
<                         linear_layer.weight.cuda(),
<                         linear_layer.weights_scaling_factor.cuda(),
<                         model_config.quantization,
<                     )
<                     print("conver to half")
<                     linear_layer.weight = mixlib.int8_matrix_to_half(linear_layer.weight.cuda()).cpu()
<                     # print("q weight is ")
<                     # print(linear_layer.weight)
<                     # print("recorver weight is ")
<                     # tmp = linear_layer.weight.to(torch.float16) * linear_layer.weights_scaling_factor[:, None].cpu()
<                     # print(tmp[0,0:10])
<                     # exit
< 
< # def pack_linear_weights(model_config: ModelConfig):
< #     """Packs the quantized linear weights in the model_config to the quantized format."""
< 
< #     def _linear_layer_to_quantized_weight(linear_layers):
< #         for linear_layer in linear_layers:
< #             if isinstance(linear_layer, LinearConfig):
< #                 if linear_layer.weights_scaling_factor is not None:
< #                     linear_layer.weight = to_quantized_weight(
< #                         linear_layer.weight,
< #                         linear_layer.weights_scaling_factor,
< #                         model_config.quantization,
< #                     )
< 
< #     if not model_config.quantization:
< #         return
< 
< #     attention_key_list = ["attention", "self_attention", "cross_attention"]
< #     for decoder_config in model_config.layers:
< #         linear_layers = []
< #         if any([hasattr(decoder_config, attention_key) for attention_key in attention_key_list]):
< #             for attention_key in attention_key_list:
< #                 attention = getattr(decoder_config, attention_key, None)
< #                 if attention:
< #                     linear_layers = [
< #                         attention.qkv,
< #                         attention.dense,
< #                     ]
< #         if decoder_config.recurrent:
< #             linear_layers = [
< #                 decoder_config.recurrent.linear_y,
< #                 decoder_config.recurrent.linear_x,
< #                 decoder_config.recurrent.linear_out,
< #             ]
< 
< #         if isinstance(decoder_config.mlp, MOEConfig):
< #             if model_config.quantization not in [QUANTIZATION_FP8, QUANTIZATION_INT4_AWQ]:
< #                 raise NotImplementedError(
< #                     f"MOE quantization for {model_config.quantization} is not supported yet."
< #                 )
< #             else:
< #                 linear_layers.append(decoder_config.mlp.experts.fc)
< #                 linear_layers.append(decoder_config.mlp.experts.proj)
< #         else:
< #             linear_layers.append(decoder_config.mlp.fc)
< #             linear_layers.append(decoder_config.mlp.proj)
< #             linear_layers.append(decoder_config.mlp.gate)
< 
< #         _linear_layer_to_quantized_weight(linear_layers)
< 
< #     if model_config.medusa_heads is not None:
< #         linear_layers = []
< 
< #         for head in model_config.medusa_heads:
< #             linear_layers.append(head.lm_head)
< #             for layer in head.medusa_layers:
< #                 linear_layers.append(layer.linear)
---
>         for head in model_config.medusa_heads:
>             linear_layers.append(head.lm_head)
>             for layer in head.medusa_layers:
>                 linear_layers.append(layer.linear)
531c431
< #         _linear_layer_to_quantized_weight(linear_layers)
---
>         _linear_layer_to_quantized_weight(linear_layers)
``` 
## 10. 文件/torch/export/model_config.py 的修改内容为：
``` 40,42d39
< QUANTIZATION_INT8_MIX = "int8_mix"
< 
< 
101,107c98
<     # for Mix precision
<     fp_ind : torch.Tensor = None
<     fp_weight : torch.Tensor = None
<     qweight : torch.Tensor = None
<     qzeros : torch.Tensor = None
<     scales : torch.Tensor = None
<     
---
> 
``` 
## 11. 文件/torch/export/tensorrt_llm_utils.py 的修改内容为：
``` 183,185c183
<     print("modelopt/torch/export/tensorrt_llm_utils.py")
<     print(model_config.quantization)
<     #exit()
---
> 
``` 
## 12. 文件/torch/export/model_config_export.py 的修改内容为：
``` 226d225
< 
263,266d261
<                     print("lianxiangTRT/modelopt/torch/export/model_config_export.py")
<                     print("----layer_config---")
<                     print(layer_config)
<                     # exit()
321,323d315
<         print("modelopt/torch/export/model_config_export.py")
<         print("-----------config.layers -----")
<         print(config.layers[0].quantization)
366,368d357
<             print("modelopt/torch/export/model_config_export.py")
<             print("checkmodel_config")
<             #print(model_config)
381,385d369
<                 print("pack_linear_weights")
<                 print("lianxiang/lianxiangTRT/modelopt/torch/export/model_config_export.py")
<                 # print(model_config)
<                 
<                 # exit()
``` 
## 13. 文件/torch/quantization/config.py 的修改内容为：
``` 246,256d245
< INT8_MIX = {
<     "quant_cfg": {
<         "*weight_quantizer": {"num_bits": (8, 16), "axis": 0},
<         "*input_quantizer": {"num_bits": 8, "axis": -1},
<         "*lm_head*": {"enable": False},
<         "*output_layer*": {"enable": False},
<         "default": {"num_bits": 8, "axis": None},
<     },
<     "algorithm": "max",
< }
< 
419,420d407
< 
< 
430d416
<     "INT8_MIX",
``` 

## 14. 文件/int8FusedDequantizeCUDA.h 的修改内容为：
``` 2,41d1
< #include <cuda_fp16.h>
< void int8FusedDequantizeCUDA(const int8_t *A,
<                              const int8_t *B,
<                              const half *scale_row,
<                              const half *scale_col,
<                              half *y, half *D, int M, int N, int K,
<                              char *,
<                              cudaStream_t);
< 
< void int8quant(int rows, int cols, const half * src, int8_t *output, 
<         half *scale,  cudaStream_t);
< 
< void print_half(const half *A, int M);
< 
< void int8dequant(int rows, int cols,  half * output, const int8_t *src,
<          const half *scale, cudaStream_t);
< 
< 
< void ExtractOutliersAndSetToZeros(int, int, const half * A, half *fp_A, const int *ind, int n, cudaStream_t);
< 
< void print_int( const  int *A, int M);
< 
< 
< void dequantizationCUDA(half * out, const int * x,
<                                  const half * scaleRow,
<                                  const half * scaleCol, int M, int N, cudaStream_t);
< 
< 
< void gemm_forward_cuda(
<     int M,
<     int N,
<     int K,
<     half *out_feats,
<     const half * in_feats, //activation
<     const int * kernel,  // int4 weight
<     const half * scaling_factors, // scaling factors
<     const int * zeros,
<     cudaStream_t stream,
<     half * cache
<     );
\ No newline at end of file
``` 
## 15. 文件/i8gemm.cu 的修改内容为：
``` 1,6d0
< #include <cuda_runtime.h>
< #include <cuda_fp16.h>
< #include "symmetric/gemm/device/gemm_dequant.h"
< #include "int8FusedDequantizeCUDA.h"
< #include <cutlass/cutlass.h>
< #include <cutlass/half.h>
8,660d1
< 
< // col 应该是 4096
< // rows 应该是 12288
< template<int size, typename T>
< __global__ void DequantKernel(const T * input,  half * output, const half * scale,
<      int rows, int cols){
< 
< 
< 
<     // each thread loads one element from global to shared mem
<     unsigned int tid = threadIdx.x;
<     unsigned int bid = blockIdx.x ;
<     if (bid > rows)
<         return ;
< 
<     const T *start = input + bid * cols;
<      __half  * d_out = output + bid * cols;
<     half quant_scales = scale[bid];
< 
<     // quant
<     for (int i = tid ; i < cols; i += size){
< 
<  
<         half tmp_ = (half) start[i];
<         d_out[i] =  static_cast<half>(( __hmul( tmp_, quant_scales ) ))  ; 
<     }
<     __syncthreads();    
< 
< }
< 
< void int8dequant(int rows, int cols,  half * output, const int8_t *src,const  half *scale,
<         cudaStream_t stream)
< {
< 
< 
<     dim3 block(256);
<     dim3 grid(rows, 1);
<     DequantKernel<256, int8_t><<<grid, block,0, stream>>>(
<                 src,
<                 output, scale,
<                 rows, cols);
< 
< }
< 
< void int8dequant(int rows, int cols,  half * output, const half *src,const  half *scale,
<             cudaStream_t stream)
< {
< 
< 
<     dim3 block(256);
<     dim3 grid(rows, 1);
<     DequantKernel<256, half><<<grid, block,0, stream>>>(
<                 src,
<                 output, scale,
<                 rows, cols);
< 
< }
< 
< template<int size>
< __global__ void FindRowScaleKernel(int8_t * output, const half * d_in, half * scale, int rows, int cols){
< 
<     __shared__ half sdata[size];
< 
<     // each thread loads one element from global to shared mem
<     unsigned int tid = threadIdx.x;
<     unsigned int bid = blockIdx.x ;
<     if (bid > rows)
<         return ;
<     const  __half *start = d_in + bid * cols;
<     int8_t * d_out = output + bid * cols;
<     sdata[tid] = __habs(start[tid]); 
<     for (int i = tid + size; i < cols; i += size)
<         sdata[tid] = __hmax ( __habs(start[i]),  sdata[tid] ); 
<     __syncthreads();
< 
< 
<     // do reduction in shared mem
<     for (unsigned int s= blockDim.x/2; s >= 1; s >>=1 ) {
<         if (tid < s) {
<             sdata[tid] =  __hmax ( __habs(sdata[tid + s]),  sdata[tid]);
<         }
<         __syncthreads();
<     }
< 
<     half  max = sdata[0];
<     // write result for this block to global mem
<     //if (tid < 32) warpReduce(sdata, tid);
< 
<     __syncthreads();
< 
<     half quant_scales = __hdiv( max, 127.0);
<     if (tid == 0){
<         scale[bid] = quant_scales;
<     }
<     // quant
<     for (int i = tid ; i < cols; i += size)
<         d_out[i] =  static_cast<int8_t>(__half2int_rn( __hdiv( start[i], quant_scales ) ))  ; 
<     __syncthreads();    
< 
< }
< 
< __global__ void PrintKernel(const half *A, int M){
< 
<     int tid = threadIdx.x;
<     float tmp = __half2float(A[tid]);
<  
<     printf("A %d is %.6f\t",tid, tmp);
< }
< void print_half(const half *A, int M){
<     dim3 block(M);
<     dim3 grid(1, 1);
<     PrintKernel<<<grid, block>>>(
<                 A, M);
< 
< }
< 
< __global__ void PrintKernelint( const  int *A, int M){
< 
<     int tid = threadIdx.x;
<     int tmp =  (A[tid]);
<  
<     printf("A %d is %d\t",tid, tmp);
< }
< void print_int(  const int *A, int M){
<     dim3 block(M);
<     dim3 grid(1, 1);
<     PrintKernelint<<<grid, block>>>(
<                 A, M);
< 
< }
< 
< void int8quant(int rows, int cols, const half * src, int8_t *output, 
<         half *scale, cudaStream_t stream){
< 
< 
<     dim3 block(256);
<     dim3 grid(rows, 1);
<     FindRowScaleKernel<256><<<grid, block, 0, stream>>>(
<                 output,
<                 src, scale,
<                 rows, cols);
< 
< }
< void int8FusedDequantizeCUDA(const int8_t *A,
<                              const int8_t *B,
<                              const half *scale_row,
<                              const half *scale_col,
<                              half *y, half *D, 
<                              int M, int N, int K,
<                              char * workspace,
<                              cudaStream_t stream) {
< 
<  
<   using Gemm = cutlass::gemm::device::symmetric::GemmDequant<
<       int8_t,                          // ElementA
<       cutlass::layout::RowMajor,       // LayoutA
<       int8_t,                          // ElementB
<       cutlass::layout::ColumnMajor,    // LayoutB
<       cutlass::half_t,                 // ElementOutput
<       cutlass::layout::RowMajor,       // LayoutOutput
<       int32_t,                         // ElementAccumulator
<       cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
<       cutlass::arch::Sm80  // tag indicating target GPU compute architecture
<       >;
< 
<   Gemm gemmOp;
<   //cutlass::Status status = gemmOp(stream);
< 
< 
<   using GemmCoord = cutlass::gemm::GemmCoord;
< 
<   typename Gemm::Arguments arguments{
<       {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
<        static_cast<GemmCoord::Index>(K)},
<       {(const int8_t *)A, K},
<       {(const int8_t *)B, K},
<       {(const cutlass::half_t *)y, N},
<       {(cutlass::half_t *)D, N},
<       {(const cutlass::half_t *)scale_col, N},
<       {(const cutlass::half_t *)scale_row, M},
<       Gemm::ElementC(1)};
< 
<   gemmOp.initialize(arguments, workspace, stream);
<   //status = gemmOp(arguments);
<   gemmOp.run(stream);
<  
< }
< 
< 
< 
< __global__  void FindOutliersAndSetToZeros_kernel(const int *ind,  half *input, 
<         half *outliersOutput, int m, int k, int len){
<  
< 
<     int tid = threadIdx.x;
<  
<     int start_col = blockIdx.x ;
<  
<     if (start_col > len)
<         return ;
< 
<   
<  
<  
<     int col = ind[start_col];
<     half *start = input +  col ;
<     half *outliersOutput_ = outliersOutput + start_col;   
<  
<     for (int i = tid; i < m ; i+=  128  ){
<         outliersOutput_[ i * len ] = start[ i * k ] ;
<         // start[ i * k ] = 0.0;
<     }
<  
<  
< 
< 
< }
< void ExtractOutliersAndSetToZeros(int M, int N, const half * A, half *fp_A, 
<         const int *ind, const int len, cudaStream_t stream){
< 
< 
<     const int blockSize = 128;
<  
< 
<     half * tmp = const_cast<half*>(A);
<     dim3 numBlocks(len);        
<     FindOutliersAndSetToZeros_kernel<<<numBlocks, blockSize, 0, 
<             stream>>>(
<             ind,
<             tmp,
<             fp_A,
<             M,
<             N,
<             len
<         );
< 
< }
< 
< 
< 
< template <typename T>
< __device__ __half convertToHalf(T value) {
<   return __int2half_rn(static_cast<int>(value));
< }
< 
< template <>
< __device__ __half convertToHalf(half value) {
<   return (__half)value;
< }
< 
< template <typename T>
< __global__ void dequantizationKernel(half *__restrict__ out,
<                                      const T *__restrict__ x,
<                                      const half *__restrict__ scaleRow,
<                                      const half *__restrict__ scaleCol,
<                                      const unsigned rows, const unsigned cols) {
<   const unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
<   const unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
<   if (col >= cols) {
<     return;
<   }
< 
<   if (row >= rows) {
<     return;
<   }
< 
<   float xElement =  static_cast<float>(x[col + row * cols]);
< 
<   out[col + row * cols] =
<       __hadd(   __float2half( ( xElement * __half2float(scaleRow[row])) * __half2float(scaleCol[col]) ) ,
<       out[col + row * cols]);
< }
< 
< 
< 
< void dequantizationCUDA(half * out, const int * x,
<                                  const half * scaleRow,
<                                  const half * scaleCol, int M, int N, cudaStream_t s) {
< 
<   unsigned rows = M;
<   unsigned cols = N;
<     
< 
< 
<   //auto out = torch::empty_like(y);
<   dim3 block{std::min<unsigned>(cols, 16),
<              std::min<unsigned>((rows - 1) + 1, 16)};
<   dim3 grid{(cols - 1) / block.x + 1, (rows - 1) / block.y + 1};
<   dequantizationKernel<<<grid, block, 0, s>>>(
<       out, x ,
<       scaleRow , scaleCol, rows, cols);
<   
< }
< 
< 
< 
< 
< 
< __device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
< {
<     uint4 result;
< 
<     uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
<     uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);
< 
<     // First, we extract the i4s and construct an intermediate fp16 number.
<     static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
<     static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
<     static constexpr uint32_t TOP_MASK              = 0x00f000f0;
<     static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;
< 
<     // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
<     // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
<     // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
<     // elt_67 to fp16 without having to shift them to the bottom bits before hand.
< 
<     // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
<     // immediately before required.
<     const uint32_t top_i4s = i4s >> 8;
<     // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
<     asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
<                     : "=r"(h[0])
<                     : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
<     // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
<     asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
<                     : "=r"(h[1])
<                     : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
<     // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
<     asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
<                     : "=r"(h[2])
<                     : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
<     // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
<     asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
<                     : "=r"(h[3])
<                     : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
< 
<     // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
<     // half2 ctor. In this case, I chose performance reliability over code readability.
< 
<     // This is the half2 {1032, 1032} represented as an integer.
<     // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
<     // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
<     static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
<     // This is the half2 {1 / 16, 1 / 16} represented as an integer.
<     static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
<     // This is the half2 {-72, -72} represented as an integer.
<     // static constexpr uint32_t NEG_72 = 0xd480d480;
<     // Haotian: Let's use {-64, -64}.
<     static constexpr uint32_t NEG_64 = 0xd400d400;
< 
<     // Finally, we construct the output numbers.
<     // Convert elt_01
<     asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
<     // Convert elt_23
<     asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
<     // Convert elt_45
<     asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
<     // Convert elt_67
<     asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
< 
<     return result;
< }
< 
< // Pack two half values.
< static inline __device__ __host__ unsigned
< __pack_half2(const half x, const half y) {
<   unsigned v0 = *((unsigned short *)&x);
<   unsigned v1 = *((unsigned short *)&y);
<   return (v1 << 16) | v0;
< }
< 
< __device__ __forceinline__ int make_divisible(int c, int divisor){
<   return (c + divisor - 1) / divisor;
< }
< 
< __global__ void __launch_bounds__(64) gemm_forward_4bit_cuda_m16n128k32(int G,
<  int split_k_iters, const half* __restrict__ A, const int* __restrict__ B,
<  const half* __restrict__ scaling_factors, const int* __restrict__ zeros,
<   int M, int IC, int OC, half* __restrict__ C) 
< {
<   static constexpr uint32_t ZERO = 0x0;
<   float C_warp[32];
<   __shared__ half A_shared[16 * (32 + 8)];
<   __shared__ half B_shared[32 * (128 + 8)];
< 
<   int j_factors1 = ((OC + 128 - 1) / 128);
<   int blockIdx_x = 0;
<   int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
<   int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);
< 
<   half A_shared_warp[8];
<   half B_shared_warp[32];
<   for (int j_0_4_init = 0; j_0_4_init < 4; ++j_0_4_init) {
<     for (int i = 0; i < 8; ++i) {
<       C_warp[(j_0_4_init * 8) + i] = 0.0;
<     }
<   }
< 
<   static constexpr int row_stride_warp = 32 * 8 / 32;
<   static constexpr int row_stride = 2 * 32 * 8 / 128;
<   bool ld_zero_flag = (threadIdx.y * 32 + threadIdx.x) * 8 < 128;
<   // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
<   bool ld_A_flag = (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;     // threadIdx.y is warp_id
<   // bool wb_C_flag = (threadIdx.x / 4) < M;
< 
<   const half* A_ptr = A 
<                 + (((int)blockIdx_y) / j_factors1 * 16 + (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) * IC
<                 + (((int)threadIdx.x) % (32 / 8)) * 8;
<   
<   const int* B_ptr = B
<             + ((int)threadIdx.y) * (OC / 8) * 2
<             + (((int)threadIdx.x) / (128 / 8)) * (OC / 8)
<             + (((int)blockIdx_y) % j_factors1) * (128 / 8)
<             + (((int)threadIdx.x) % (128 / 8)) * 1;
< // Why * 1 in the above line?
<                         
<   half* A_shared_ptr = A_shared 
<                     + ((int)threadIdx.y) * row_stride_warp * (32 + 8) 
<                     + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
<                     + (((int)threadIdx.x) % (32 / 8) ) * 8;
< 
<   half* B_shared_ptr = B_shared
<                     + ((int)threadIdx.y) * (row_stride / 2) * (128 + 8)
<                     + (((int)threadIdx.x) / (128 / 8)) * (128 + 8)
<                     + (((int)threadIdx.x) % (128 / 8)) * 8;
<   
<   const int* zeros_ptr = zeros
<                 + (((int)blockIdx_y) % j_factors1) * (128 / 8)
<                 + ((int)threadIdx.x) % (128 / 8);
<   
<   const half* scaling_factors_ptr = scaling_factors
<                             + (((int)blockIdx_y) % j_factors1) * (128) 
<                             + (((int)threadIdx.x) % (128 / 8)) * 8;
< 
<   half* C_ptr = C 
<               + static_cast<long long>(blockIdx_z) * M * OC        // blockIdz.x -> split_k dim
<               + (((int)blockIdx_y) % j_factors1) * 128
<               + ((int)threadIdx.y) * 64
<               + (((int)threadIdx.x) % 4) * 2;
< 
<   // preload s.f. and zeros
<   int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
<   if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
<   for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
<     int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
<     __syncthreads();
<     // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
<     if (ld_A_flag)
<     {
<       *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
<     }
<     else
<     {
<       *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
<     }
< 
<     // for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
<     uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
<     uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
<     uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
<     /*
<     if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 && threadIdx.y == 0){
<       printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x, B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x, B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
<     }
<     */
<     // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
<     const int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);
< 
<     for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {
< 
<       // B: 32 x 136 (128+8) float16
<       // each warp: 32 x 4
<       // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus zero -> WB UINT4
<       // *(uint4*)(B_shared + ((((ax0_ax1_fused_0 * 544) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(B + ((((((k_0_0 * 163840) + (ax0_ax1_fused_0 * 20480)) + (((int)threadIdx.y) * 10240)) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx_y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
<       // row stride in shared memory: (NWARPS * 32 * 8 / cta_N) 
<       uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
<       uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
<       //uint4 B_loaded_zero = *(uint4*)(zeros_shared + (threadIdx.x % (cta_N / 8)) * 8);
< 
<       // uint4 B_loaded_scale = *(uint4*)(scaling_factors_shared + (threadIdx.x % (cta_N / 8)) * 8);
<       // - zero and * scale
<       // TODO (Haotian): can save 4 assembly instructions if sormulate as deq = q * scale - zero * scale.
<       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
<       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
<       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
<       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
<       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
<       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
<       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
<       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
<       /*
<       if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 17 && threadIdx.y == 0){
<         printf("[x] %X %X %X %X\n", B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
<       }
<       */
< 
<       // write back
<       *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
<     }
<     __syncthreads();
< 
<     for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
<       {
<         unsigned int addr;
<         asm volatile(
<           "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
<           : "=r"(addr)
<           : "l"((void *)((&(A_shared[(k_0_1 * 16)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
<         );
< 
< 
<         asm volatile(
<           "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
<           "{%0, %1, %2, %3}, [%4];\n"
<           : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
<           : "r"(addr)
<         );
<       }
< 
<       for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
<         {
<           unsigned int addr;
<           asm volatile(
<             "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
<             : "=r"(addr)
<             : "l"((void *)((&(B_shared[(((k_0_1 * 2176) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 136) + ((((int)threadIdx.x) >> 4) * 8))))
<           );
<           asm volatile(
<             "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
<             "{%0, %1, %2, %3}, [%4];\n"
<             : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
<             : "r"(addr)
<           );
<         }
<       }
<       for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
< #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
<         {
<           asm volatile(
<             "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
<             "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
<             :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
<             : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
<         }
< 
<         {
<           asm volatile(
<             "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
<             "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
<             :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
<             : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
<         }
< 
<         {
<           asm volatile(
<             "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
<             "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
<             :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
<             : "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
<         }
< 
<         {
<           asm volatile(
<             "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
<             "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
<             :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
<             : "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
<         }
< #else
<         {
<           asm volatile(
<             "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
<             "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
<             :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
<             : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
<         }
< 
<         {
<           asm volatile(
<             "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
<             "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
<             :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
<             : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
<         }
< #endif
<       }
<     }
<   }
< 
< // TODO: Shang: Hoist loop invariance.
<   for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
<     for (int local_id = 0; local_id < 8; ++local_id) {
<       int row_offset = (((int)blockIdx_y) / j_factors1) * 16 + ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
<       if (row_offset < M)
<       {
<         *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
<       }
<     }
<   }
< }
< 
< __global__ void sum_all(const half * source, half *out, int M, int N){
< 
<   int tid = threadIdx.x;
<   int row = blockIdx.x;
<   int col = blockIdx.y * 128 + tid;
<  
<   assert ( col < N);
<   assert ( row < M);
< 
<   const half * src =  source + (row * N  + col) ; //先要定位到 row 和 col 然后求和！！！
<   half *dest = out + ( row * N + col);
<   half tmp = 0.0;
<   for (int i = 0 ; i < 8; ++i){
<       tmp = __hadd(src[ ( M * N  ) * i ], tmp);
<   }
<   dest[0] = tmp;
< }
< void gemm_forward_cuda(
<     int M,
<     int N,
<     int K,
<     half *out_feats,
<     const half * in_feats, //activation
<     const int * kernel,  // int4 weight
<     const half * scaling_factors, // scaling factors
<     const int * zeros,
<     cudaStream_t stream,
<     half * tmp
<     ){
< 
<     int num_in_feats = M;
<     int num_in_channels = K;
<     const int group_size = 128;
<     const int split_k_iters = 8;
<     int num_out_feats = M;
<     int num_out_channels = N;  //12288 N
<  
<     int j_factors1 = num_out_channels / 128 / 1;
<     dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);
<     // threadIdx.x: 32
<     // threadIdx.y: i_factors[2] * j_factors[2]
<     dim3 threads_per_block(32, 2);
<     gemm_forward_4bit_cuda_m16n128k32<<<num_blocks, threads_per_block, 0, stream>>>(
<         group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats,
<          num_in_channels, num_out_channels, tmp);
< 
<     assert ( M < 65536 );
<     assert ( (N % 128) == 0 );
<     dim3 num_blocks_sum( M, ( N / 128 ) );
<     sum_all<<<num_blocks_sum, 128, 0, stream>>>(tmp, out_feats, M, N);
< 
< }
\ No newline at end of file
``` 
## 16. 文件/symmetric/epilogue/threadblock/default_epilogue_tensor_op_dequant.h 的修改内容为：
``` 1,32d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< #pragma once
34,87d1
< #include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
< #include "symmetric/epilogue/threadblock/epilogue_dequant.h"
< #include "symmetric/epilogue/threadblock/predicated_vcol_iterator.h"
< #include "symmetric/epilogue/threadblock/predicated_vrow_iterator.h"
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< namespace epilogue {
< namespace threadblock {
< namespace symmetric {
< ////////////////////////////////////////////////////////////////////////////////
< template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
<           typename OutputOp_, int ElementsPerAccess, bool ScatterD = false,
<           typename PermuteDLayout = layout::NoPermute>
< struct DefaultEpilogueTensorOpDequant
<     : public DefaultEpilogueTensorOp<Shape_, WarpMmaTensorOp_, PartitionsK,
<                                      OutputOp_, ElementsPerAccess, ScatterD,
<                                      PermuteDLayout> {
<   using OutputOp = OutputOp_;
<   using DefaultEpilogueTensorOp =
<       DefaultEpilogueTensorOp<Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_,
<                               ElementsPerAccess, ScatterD, PermuteDLayout>;
<   using RowVecIterator =
<       cutlass::epilogue::threadblock::symmetric::PredicatedVRowIterator<
<           typename DefaultEpilogueTensorOp::OutputTileThreadMap,
<           typename DefaultEpilogueTensorOp::ElementOutput, ScatterD,
<           PermuteDLayout, DefaultEpilogueTensorOp::UseCUDAStore>;
<   using ColVecIterator =
<       cutlass::epilogue::threadblock::symmetric::PredicatedVColIterator<
<           typename DefaultEpilogueTensorOp::OutputTileThreadMap,
<           typename DefaultEpilogueTensorOp::ElementOutput, ScatterD,
<           PermuteDLayout, DefaultEpilogueTensorOp::UseCUDAStore>;
< 
<   using Epilogue = cutlass::epilogue::threadblock::symmetric::EpilogueDequant<
<       typename DefaultEpilogueTensorOp::Shape,
<       typename DefaultEpilogueTensorOp::WarpMmaTensorOp,
<       DefaultEpilogueTensorOp::kPartitionsK,
<       typename DefaultEpilogueTensorOp::OutputTileIterator, RowVecIterator,
<       ColVecIterator,
<       typename DefaultEpilogueTensorOp::AccumulatorFragmentIterator,
<       typename DefaultEpilogueTensorOp::WarpTileIterator,
<       typename DefaultEpilogueTensorOp::SharedLoadIterator, OutputOp,
<       typename DefaultEpilogueTensorOp::Padding,
<       DefaultEpilogueTensorOp::kFragmentsPerIteration>;
< };
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< }  // namespace symmetric
< }  // namespace threadblock
< }  // namespace epilogue
< }  // namespace cutlass
< 
< ////////////////////////////////////////////////////////////////////////////////
``` 
## 17. 文件/symmetric/epilogue/threadblock/predicated_vrow_iterator.h 的修改内容为：
``` 1,33d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< /*! \file
<   \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.
35,470d1
<   The epilogue rearranges the result of a matrix product through shared memory
<   to match canonical tensor layouts in global memory. Epilogues support
<   conversion and reduction operations.
< 
< */
< 
< #pragma once
< 
< #include "cutlass/arch/arch.h"
< #include "cutlass/arch/memory.h"
< #include "cutlass/array.h"
< #include "cutlass/cutlass.h"
< #include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
< #include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
< #include "cutlass/layout/matrix.h"
< #include "cutlass/layout/permute.h"
< #include "cutlass/layout/tensor.h"
< #include "cutlass/matrix_shape.h"
< #include "cutlass/numeric_types.h"
< #include "cutlass/tensor_ref.h"
< #include "cutlass/transform/pitch_linear_thread_map.h"
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace epilogue {
< namespace threadblock {
< namespace symmetric {
< ////////////////////////////////////////////////////////////////////////////////
< 
< /// Tile iterator used to load and store output tile from global memory in
< /// epilogue.
< ///
< /// Satisfies: ReadableTileIterator | PredicatedTileIterator |
< /// ForwardTileIterator
< ///
< template <typename ThreadMap_,    ///< Thread map (conept: OutputTileThreadMap)
<           typename Element_,      ///< Element data type
<           bool ScatterD = false,  ///< Scatter D operand or not
<           typename PermuteDLayout =
<               layout::NoPermute,  ///< Permute D operand or not
<           bool UseCUDAStore = false>
< class PredicatedVRowIterator {
<  public:
<   using ThreadMap = ThreadMap_;
<   using Shape = typename ThreadMap::Shape;
< 
<   using Element = Element_;
< 
<   using Layout = layout::RowMajor;
<   using TensorRef = TensorRef<Element, Layout>;
<   using ConstTensorRef = typename TensorRef::ConstTensorRef;
< 
<   using Index = typename Layout::Index;
<   using LongIndex = typename Layout::LongIndex;
<   using TensorCoord = MatrixCoord;
< 
<   static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
<   static int const kThreads = ThreadMap::kThreads;
<   static int const kIterations = ThreadMap::Count::kTile;
< 
<   static bool constexpr PermuteD = !layout::is_trivial_permute<PermuteDLayout>;
< 
<   static_assert(ThreadMap::Iterations::kRow > 0,
<                 "ThreadMap::Iterations::kRow must be > 0");
<   static_assert(ThreadMap::Iterations::kGroup > 0,
<                 "ThreadMap::Iterations::kGroup must be > 0");
<   static_assert(ThreadMap::Iterations::kCluster > 0,
<                 "ThreadMap::Iterations::kCluster must be > 0");
<   static_assert(ThreadMap::Iterations::kColumn > 0,
<                 "ThreadMap::Iterations::kColumn must be > 0");
< 
<   /// Fragment object
<   using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
<                                       ThreadMap::Iterations::kRow *
<                                       ThreadMap::Iterations::kGroup *
<                                       ThreadMap::Iterations::kCluster *
<                                       ThreadMap::kElementsPerAccess>;
<   /// Memory access size
<   using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;
< 
<   //
<   // Parameters struct
<   //
< 
<   /// Uses a non-template class
<   struct Params : PredicatedTileIteratorParams {
<     using Base = PredicatedTileIteratorParams;
< 
<     CUTLASS_HOST_DEVICE
<     Params() {}
< 
<     CUTLASS_HOST_DEVICE
<     Params(Layout const &layout)
<         : PredicatedTileIteratorParams(
<               0, make_OutputTileThreadMapDesc<ThreadMap>()) {}
< 
<     CUTLASS_HOST_DEVICE
<     Params(Base const &base) : Base(base) {}
<   };
< 
<   /// Mask object
<   struct Mask {
<     static int const kCount = ThreadMap::Iterations::kColumn;
< 
<     /// Predicate state
<     bool predicates[kCount];
< 
<     //
<     // Mask
<     //
<     CUTLASS_HOST_DEVICE
<     Mask() { enable(); }
< 
<     ///< Efficiently disables all accesses guarded by mask
<     CUTLASS_HOST_DEVICE void clear() {
<       CUTLASS_PRAGMA_UNROLL
<       for (int i = 0; i < kCount; ++i) {
<         predicates[i] = false;
<       }
<     }
< 
<     ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
<     CUTLASS_DEVICE void enable() {
<       CUTLASS_PRAGMA_UNROLL
<       for (int i = 0; i < kCount; ++i) {
<         predicates[i] = true;
<       }
<     }
<   };
< 
<  private:
<   //
<   // Data members
<   //
< 
<   /// Parameters structure containing reference and precomputed state.
<   PredicatedTileIteratorParams params_;
< 
<   /// Byte-level pointer. This pointer is usually for both load() and store(),
<   /// unless PermuteD is performed. When having PermuteD, byte_pointer_ is only
<   /// for load().
<   uint8_t *byte_pointer_;
< 
<   /// Byte-level pointer for store(). Due to PermuteD Op, store_byte_pointer_
<   /// may be with different address computation compared to byte_pointer_.
<   uint8_t *store_byte_pointer_;
< 
<   /// Array of boolean values to contain steady-state predicates
<   Mask mask_;
< 
<   /// Extent of the matrix tile in rows
<   Index extent_row_;
< 
<   /// Extent of the matrix tile in rows
<   Index extent_column_;
< 
<   /// A thread's starting row position (assuming steady-state predicates have
<   /// been computed)
<   Index thread_start_row_;
< 
<   /// A thread's starting column
<   Index thread_start_column_;
< 
<   /// Internal state counter
<   int state_[3];
< 
<   /// Scatter indices
<   int const *indices_;
< 
<   /// PermuteDLayout
<   PermuteDLayout permute_layout_;
< 
<   //
<   // Static asserts about internal strides
<   //
< 
<   static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
<   static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
<   static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8,
<                 "Expected 64b strides");
< 
<  private:
<   //
<   // Methods
<   //
< 
<  public:
<   //
<   // Methods
<   //
< 
<   /// Constructor
<   CUTLASS_DEVICE
<   PredicatedVRowIterator(PredicatedTileIteratorParams const &params,
<                          Element *pointer, TensorCoord extent, int thread_idx,
<                          TensorCoord threadblock_offset = TensorCoord(),
<                          int const *indices = nullptr)
<       : params_(params),
<         indices_(indices),
<         permute_layout_(
<             PitchLinearCoord(extent.column(), extent.row()),
<             params_.stride * kElementsPerAccess / sizeof(AccessType)) {
<     TensorCoord thread_offset =
<         ThreadMap::initial_offset(thread_idx) + threadblock_offset;
< 
<     extent_row_ = extent.row();
<     extent_column_ = extent.column();
< 
<     thread_start_row_ = thread_offset.row();
<     thread_start_column_ = thread_offset.column();
< 
<     // Initialize predicates
<     CUTLASS_PRAGMA_UNROLL
<     for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
<       mask_.predicates[c] = ((thread_offset.column() +
<                               ThreadMap::Delta::kColumn * c) < extent.column());
<     }
< 
<     // Null pointer performs no accesses
<     if (!pointer) {
<       mask_.clear();
<     }
< 
<     if (ScatterD && !indices) {
<       mask_.clear();
<     }
< 
<     // Initialize byte_pointer_
<     byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
<                     LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
<                     LongIndex(thread_offset.column()) * sizeof(AccessType) /
<                         kElementsPerAccess;
< 
<     if (ScatterD) {
<       byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
<                       LongIndex(thread_offset.column()) * sizeof(AccessType) /
<                           kElementsPerAccess;
<     }
< 
<     // store_byte_pointer_ is set to be the same with byte_pointer_ unless
<     // PermuteD is used.
<     store_byte_pointer_ =
<         PermuteD ? reinterpret_cast<uint8_t *>(pointer) : byte_pointer_;
< 
<     // Initialize internal state counter
<     state_[0] = state_[1] = state_[2] = 0;
<   }
< 
<   /// Adds a pointer offset in units of Element
<   CUTLASS_HOST_DEVICE
<   void add_pointer_offset(LongIndex pointer_offset) {
<     store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
<     byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
<   }
< 
<   /// Loads a fragment from memory
<   CUTLASS_DEVICE
<   void load_with_byte_offset(Fragment &frag, int64_t byte_offset) const {
<     uint8_t *byte_pointer = byte_pointer_;
<     AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
< 
<     CUTLASS_PRAGMA_UNROLL
<     for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
<          ++cluster) {
<       CUTLASS_PRAGMA_UNROLL
<       for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
<         CUTLASS_PRAGMA_UNROLL
<         for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
<           int frag_row_idx =
<               (row + ThreadMap::Iterations::kRow *
<                          (group + ThreadMap::Iterations::kGroup * cluster));
< 
<           int row_offset = row * ThreadMap::Delta::kRow +
<                            group * ThreadMap::Delta::kGroup +
<                            cluster * ThreadMap::Delta::kCluster;
< 
<           bool row_guard = ((row_offset + thread_start_row_) < extent_row_);
< 
<           AccessType *memory_pointer =
<               reinterpret_cast<AccessType *>(byte_pointer + byte_offset);
< 
<           if (ScatterD && row_guard) {
<             assert(indices_);
< 
<             memory_pointer = reinterpret_cast<AccessType *>(
<                 byte_pointer + byte_offset +
<                 LongIndex(indices_[row_offset + thread_start_row_]) *
<                     LongIndex(params_.stride));
<           }
< 
<           CUTLASS_PRAGMA_UNROLL
<           for (int column = 0; column < ThreadMap::Iterations::kColumn;
<                ++column) {
<             bool guard = row_guard && mask_.predicates[column];
< 
<             cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
<                 frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
<                          column],
<                 (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
<                                         kElementsPerAccess],
<                 guard);
<           }
< 
<           if (row + 1 < ThreadMap::Iterations::kRow) {
<             if (!ScatterD) {
<               byte_pointer += params_.increment_row;
<             }
<           }
<         }
< 
<         if (group + 1 < ThreadMap::Iterations::kGroup) {
<           byte_pointer += params_.increment_group;
<         }
<       }
< 
<       if (cluster + 1 < ThreadMap::Iterations::kCluster) {
<         byte_pointer += params_.increment_cluster;
<       }
<     }
<   }
< 
<   /// Loads a fragment from memory
<   CUTLASS_DEVICE
<   void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }
< 
<   CUTLASS_DEVICE
<   MatrixCoord thread_start() const {
<     return MatrixCoord(thread_start_row_, thread_start_column_);
<   }
< 
<   /// Need to get the thread start row from the tile iterator
<   CUTLASS_DEVICE
<   int32_t thread_start_row() const { return thread_start_row_; }
< 
<   /// Need to get the thread start row from the tile iterator
<   CUTLASS_DEVICE
<   int32_t thread_start_column() const { return thread_start_column_; }
< 
<   /// Extent of the matrix in rows
<   CUTLASS_DEVICE
<   Index extent_row() const { return extent_row_; }
< 
<   /// Extent of the matrix in columns
<   CUTLASS_DEVICE
<   Index extent_column() const { return extent_column_; }
< 
<   /// Advances to the next position to load or store
<   CUTLASS_HOST_DEVICE
<   PredicatedVRowIterator &operator++() {
<     ++state_[0];
< 
<     if (!ScatterD) {
<       byte_pointer_ += params_.advance_row;
<     }
< 
<     if (!ScatterD && !PermuteD) {
<       store_byte_pointer_ += params_.advance_row;
<     }
< 
<     thread_start_row_ += ThreadMap::Shape::kRow;
< 
<     if (state_[0] == ThreadMap::Count::kRow) {
<       state_[0] = 0;
<       ++state_[1];
< 
<       if (!ScatterD) {
<         byte_pointer_ += params_.advance_group;
<       }
< 
<       if (!ScatterD && !PermuteD) {
<         store_byte_pointer_ += params_.advance_group;
<       }
< 
<       thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
<                            ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
< 
<       if (state_[1] == ThreadMap::Count::kGroup) {
<         state_[1] = 0;
<         ++state_[2];
< 
<         if (!ScatterD) {
<           byte_pointer_ += params_.advance_cluster;
<         }
< 
<         if (!ScatterD && !PermuteD) {
<           store_byte_pointer_ += params_.advance_cluster;
<         }
< 
<         thread_start_row_ += ThreadMap::Count::kGroup *
<                              ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
<                              ThreadMap::Shape::kRow;
< 
<         if (state_[2] == ThreadMap::Count::kCluster) {
<           state_[2] = 0;
< 
<           if (!ScatterD) {
<             byte_pointer_ += params_.advance_tile;
<           }
< 
<           if (!ScatterD && !PermuteD) {
<             store_byte_pointer_ += params_.advance_tile;
<           }
< 
<           thread_start_row_ +=
<               ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
<               ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
<         }
<       }
<     }
< 
<     return *this;
<   }
< 
<   ///< Efficiently disables all accesses guarded by mask
<   CUTLASS_DEVICE void clear_mask() { mask_.clear(); }
< 
<   ///< Efficiently enables all accesses guarded by mask
<   CUTLASS_DEVICE void enable_mask() { mask_.enable(); }
< 
<   ///< Sets the mask
<   CUTLASS_DEVICE void get_mask(Mask &mask) const { mask = mask_; }
< 
<   ///< Sets the mask
<   CUTLASS_DEVICE void set_mask(Mask const &mask) { mask_ = mask; }
< };
< 
< }  // namespace symmetric
< }  // namespace threadblock
< }  // namespace epilogue
< }  // namespace cutlass
< 
< ////////////////////////////////////////////////////////////////////////////////
\ No newline at end of file
``` 
## 18. 文件/symmetric/epilogue/threadblock/epilogue_dequant.h 的修改内容为：
``` 1,33d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< /*! \file
<   \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.
35,448d1
<   The epilogue rearranges the result of a matrix product through shared memory
<   to match canonical tensor layouts in global memory. Epilogues support
<   conversion and reduction operations.
< 
<   The shared memory resource is time-sliced across warps.
< */
< 
< #pragma once
< 
< #if defined(__CUDACC_RTC__)
< #include <cuda/std/cassert>
< #else
< #include <assert.h>
< #endif
< 
< #include "cutlass/aligned_buffer.h"
< #include "cutlass/array.h"
< #include "cutlass/cutlass.h"
< #include "cutlass/epilogue/threadblock/epilogue_base.h"
< #include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
< #include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
< #include "cutlass/functional.h"
< #include "cutlass/gemm/gemm.h"
< #include "cutlass/layout/tensor.h"
< #include "cutlass/layout/vector.h"
< #include "cutlass/numeric_types.h"
< #include "cutlass/tensor_coord.h"
< #include "cutlass/transform/pitch_linear_thread_map.h"
< #include "cutlass/transform/threadblock/regular_tile_iterator.h"
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< namespace epilogue {
< namespace threadblock {
< namespace symmetric {
< ////////////////////////////////////////////////////////////////////////////////
< 
< /// Epilogue operator
< template <typename Shape_,  ///< Shape of threadblock tile (concept: GemmShape)
<           typename WarpMmaOperator_,  ///< Warp-level MMA operator (concept:
<                                       ///< gemm::warp::MmaTensorOp)
<           int PartitionsK,  ///< Number of partitions of the K dimension
<           typename OutputTileIterator_,  ///< Tile iterator reading and writing
<                                          ///< output tensors
<           typename RowVecIterator_,      ///< Row broadcast iterator reading row
<                                          ///< vector tensors
<           typename ColVecIterator_,      ///< Col broadcast iterator reading col
<                                          ///< vector tensors
<           typename AccumulatorFragmentIterator_,  ///< Fragment iterator
<                                                   ///< selecting accumulators
<           typename WarpTileIterator_,    ///< Warp-scoped tile iterator writing
<                                          ///< accumulators to SMEM
<           typename SharedLoadIterator_,  ///< Threadblock-scoped tile iterator
<                                          ///< loading from SMEM
<           typename OutputOp_,            ///< Output operator
<           typename Padding_,  ///< Padding added to SMEM allocation to avoid
<                               ///< bank conflicts (concept: MatrixShape)
<           int FragmentsPerPartition =
<               1,                  ///< Used to coarsten the epilogue granularity
<           int IterationsUnroll =  ///< Used to reduce binary size when epilogue
<                                   ///< op is large
<           (!IsEpilogueFunctorHeavy<OutputOp_>::value)>
< class EpilogueDequant
<     : public EpilogueBase<Shape_, typename WarpMmaOperator_::Shape, PartitionsK,
<                           AccumulatorFragmentIterator_, WarpTileIterator_,
<                           Padding_, FragmentsPerPartition>,
<       public EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_,
<                                  AccumulatorFragmentIterator_> {
<  public:
<   using Base = EpilogueBase<Shape_, typename WarpMmaOperator_::Shape,
<                             PartitionsK, AccumulatorFragmentIterator_,
<                             WarpTileIterator_, Padding_, FragmentsPerPartition>;
< 
<   using BaseStreamK = EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_,
<                                           AccumulatorFragmentIterator_>;
< 
<   using Shape = Shape_;
<   using WarpMmaOperator = WarpMmaOperator_;
<   static int const kPartitionsK = PartitionsK;
<   using OutputTileIterator = OutputTileIterator_;
<   using RowVecIterator = RowVecIterator_;
<   using ColVecIterator = ColVecIterator_;
<   using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
<   using WarpTileIterator = WarpTileIterator_;
<   using SharedLoadIterator = SharedLoadIterator_;
<   using OutputOp = OutputOp_;
<   using Padding = Padding_;
<   using Layout = layout::RowMajor;
<   using LongIndex = typename Layout::LongIndex;
< 
<   /// Number of warps per block
<   using WarpCount = typename Base::WarpCount;
< 
<   /// Number of threads per block
<   static int const kBlockThreads = 32 * WarpCount::kCount;
< 
<   /// Per-thread accumulator tile type
<   using AccumulatorTile = typename Base::AccumulatorTile;
< 
<   /// Numerical accumulation element type
<   using ElementAccumulator = typename WarpMmaOperator::ElementC;
< 
<   /// Fragment type used by the accumulator tile's fragment iterator
<   using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;
< 
<   /// Output element
<   using ElementOutput = typename OutputTileIterator::Element;
< 
<   /// Output access size
<   static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
< 
<   /// Tensor reference to destination tensor
<   using TensorRef = typename OutputTileIterator::TensorRef;
< 
<   /// Tensor reference to sync tensor
<   using SyncTensorRef =
<       typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;
< 
<   /// Const tensor reference to source tensor
<   using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;
< 
<   /// Vector type used by the global output iterator
<   using OutputAccessType = Array<typename OutputTileIterator::Element,
<                                  OutputTileIterator::kElementsPerAccess>;
< 
<   /// Vector type used by the shared output iterator
<   using AccumulatorAccessType = Array<typename WarpTileIterator::Element,
<                                       OutputTileIterator::kElementsPerAccess>;
< 
<   static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1
<                                         ? Base::kFragmentsPerIteration
<                                         : kPartitionsK;
< 
<   static int constexpr kSmemPointerOffset =
<       Base::SharedStorage::StorageShape::kCount / kSmemTiles;
< 
<  public:
<   static_assert(
<       SharedLoadIterator::Fragment::kElements ==
<           OutputTileIterator::Fragment::kElements,
<       "Mismatch between shared load iterator and output tile iterator.");
< 
<   static_assert(OutputTileIterator::kElementsPerAccess,
<                 "OutputTileIterator::kElementsPerAccess must not be zero.");
< 
<   static_assert(!(OutputTileIterator::Fragment::kElements %
<                   OutputTileIterator::kElementsPerAccess),
<                 "Divisibility");
< 
<   static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1,
<                 "One of these must be exactly 1.");
< 
<  public:
<   /// Aspect for when epilogue source is needed
<   struct SourceAspectNeeded {
<     OutputTileIterator source_iterator;
<     RowVecIterator row_vec_iterator;
<     ColVecIterator col_vec_iterator;
< 
<     typename OutputTileIterator::Fragment source_fragment;
<     typename RowVecIterator::Fragment row_vec_fragment;
<     typename ColVecIterator::Fragment col_vec_fragment;
< 
<     /// Invoke the output functor over each vector of output
<     CUTLASS_DEVICE
<     static void apply_output_operator(
<         typename OutputTileIterator::Fragment &output_fragment,
<         OutputOp const &output_op,
<         typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
<         typename OutputTileIterator::Fragment const &source_fragment,
<         typename RowVecIterator::Fragment const &row_vec_fragment,
<         typename ColVecIterator::Fragment const &col_vec_fragment) {
<       OutputAccessType *output_frag_ptr =
<           reinterpret_cast<OutputAccessType *>(&output_fragment);
< 
<       AccumulatorAccessType const *compute_frag_ptr =
<           reinterpret_cast<AccumulatorAccessType const *>(
<               &aligned_accum_fragment);
< 
<       OutputAccessType const *source_frag_ptr =
<           reinterpret_cast<OutputAccessType const *>(&source_fragment);
< 
<       OutputAccessType const *row_vec_frag_ptr =
<           reinterpret_cast<OutputAccessType const *>(&row_vec_fragment);
< 
<       OutputAccessType const *col_vec_frag_ptr =
<           reinterpret_cast<OutputAccessType const *>(&col_vec_fragment);
< 
<       int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
<                                       OutputTileIterator::kElementsPerAccess;
< 
<       CUTLASS_PRAGMA_UNROLL
<       for (int i = 0; i < kOutputOpIterations; ++i) {
<         // Call the output operator
<         output_frag_ptr[i] =
<             output_op(compute_frag_ptr[i], source_frag_ptr[i],
<                       row_vec_frag_ptr[i], col_vec_frag_ptr[i]);
<       }
<     }
< 
<     /// Constructor
<     CUTLASS_DEVICE
<     SourceAspectNeeded(OutputTileIterator source_iterator,
<                        RowVecIterator row_vec_iterator,
<                        ColVecIterator col_vec_iterator)
<         : source_iterator(source_iterator),
<           row_vec_iterator(row_vec_iterator),
<           col_vec_iterator(col_vec_iterator) {
<       source_fragment.clear();
<       row_vec_fragment.clear();
<       col_vec_fragment.clear();
<     }
< 
<     // Load addend source fragment from global memory
<     CUTLASS_DEVICE
<     void load() {
<       source_iterator.load(source_fragment);
<       row_vec_iterator.load(row_vec_fragment);
<       col_vec_iterator.load(col_vec_fragment);
<       ++source_iterator;
<       ++row_vec_iterator;
<       ++col_vec_iterator;
<     }
< 
<     /// Invoke the output functor over each vector of output
<     CUTLASS_DEVICE
<     void apply_output_operator(
<         typename OutputTileIterator::Fragment &output_fragment,
<         OutputOp const &output_op,
<         typename SharedLoadIterator::Fragment const &aligned_accum_fragment) {
<       apply_output_operator(output_fragment, output_op, aligned_accum_fragment,
<                             source_fragment, row_vec_fragment,
<                             col_vec_fragment);
<     }
<   };
< 
<  private:
<   /// Loads fragment from shared memory aligned with output tensor
<   SharedLoadIterator shared_load_iterator_;
< 
<   /// Thread index in the threadblock
<   int thread_idx;
< 
<   /// Warp index in the threadblock
<   int warp_idx;
< 
<  public:
<   /// Constructor
<   CUTLASS_DEVICE
<   EpilogueDequant(
<       typename Base::SharedStorage &shared_storage,  ///< Shared storage object
<       int thread_idx,  ///< ID of a thread within the threadblock
<       int warp_idx,    ///< ID of warp within threadblock
<       int lane_idx)    ///< Id of thread within warp
<       : Base(shared_storage, thread_idx, warp_idx, lane_idx),
<         BaseStreamK(thread_idx),
<         shared_load_iterator_(shared_storage.reference(), thread_idx),
<         thread_idx(thread_idx),
<         warp_idx(warp_idx) {}
< 
<   /// Perform the epilogue computations and stream the result to global memory.
<   /// Implements two alternative codepaths, depending on whether the output op
<   /// requires addend data to be loaded.
<   CUTLASS_DEVICE
<   void operator()(
<       OutputOp const &output_op,  ///< Output operator
<       OutputTileIterator
<           destination_iterator,  ///< Tile iterator for destination
<       AccumulatorTile const
<           &accumulators,  ///< Complete warp-level accumulator tile
<       OutputTileIterator source_iterator,  ///< Tile iterator for addend source
<       RowVecIterator row_vec_iterator,  ///< Vector iterator for addend source
<       ColVecIterator col_vec_iterator)  ///< Vector iterator for addend source
<   {
<     operator()(output_op, destination_iterator, accumulators,
<                SourceAspectNeeded(source_iterator, row_vec_iterator,
<                                   col_vec_iterator));
<   }
< 
<   /// Perform the epilogue computations and stream the result to global memory.
<   /// Implements a single codepath, regardless of whether the output op requires
<   /// addend data to be loaded
<   CUTLASS_DEVICE
<   void unified(
<       OutputOp const &output_op,  ///< Output operator
<       OutputTileIterator
<           destination_iterator,  ///< Tile iterator for destination
<       AccumulatorTile const
<           &accumulators,  ///< Complete warp-level accumulator tile
<       OutputTileIterator source_iterator)  ///< Tile iterator for addend source
<   {
<     if (!output_op.is_source_needed()) {
<       source_iterator.clear_mask();
<       __syncthreads();  // Dummy (CUDA 11.0)
<     }
< 
<     operator()(output_op, destination_iterator, accumulators,
<                SourceAspectNeeded(source_iterator));
<   }
< 
<   template <class Seq>
<   struct acc2smem;
< 
<   template <size_t... Seq>
<   struct acc2smem<cutlass::index_sequence<Seq...>> {
<     template <int Advance>
<     CUTLASS_DEVICE static void helper(
<         AccumulatorFragmentIterator accum_fragment_iterator,
<         WarpTileIterator &warp_tile_iterator) {
<       CUTLASS_PRAGMA_UNROLL
<       for (int i = 0; i < Advance; i++) {
<         ++accum_fragment_iterator;
<       }
< 
<       typename AccumulatorFragmentIterator::Fragment accum_fragment;
< 
<       accum_fragment_iterator.load(accum_fragment);
<       ++accum_fragment_iterator;
<       warp_tile_iterator.store(accum_fragment);
<     }
< 
<     CUTLASS_DEVICE
<     static void push(size_t pos,
<                      AccumulatorFragmentIterator const &iterator_begin,
<                      WarpTileIterator &warp_tile_iterator) {
<       int dummy[] = {(pos == Seq) &&
<                      (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
<     }
<   };
< 
<   /// Streams the result to global memory
<   template <typename SourceAspect>
<   CUTLASS_DEVICE void operator()(
<       OutputOp const &output_op,  ///< Output operator
<       OutputTileIterator
<           destination_iterator,  ///< Tile iterator for destination
<       AccumulatorTile const
<           &accumulators,  ///< Complete warp-level accumulator tile
<       SourceAspect source) {
<     // Iterator over warp-level accumulator fragment
<     AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
< 
<     //
<     // Iterate over accumulator tile
<     //
< 
< #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
<     for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
<       //
<       // Load the source
<       //
< 
<       source.load();
<       //
<       // Convert and store fragment
<       //
< 
<       __syncthreads();
< 
<       acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::
<           push(iter, accum_fragment_iterator, this->warp_tile_iterator_);
< 
<       __syncthreads();
< 
<       //
<       // Load fragments from shared memory
<       //
< 
<       typename SharedLoadIterator::Fragment
<           aligned_accum_fragment[kPartitionsK];
<       shared_load_iterator_.load(aligned_accum_fragment[0]);
< 
<       if (kPartitionsK > 1) {
<         plus<typename SharedLoadIterator::Fragment> add_fragments;
< 
<         CUTLASS_PRAGMA_UNROLL
<         for (int i = 1; i < kPartitionsK; ++i) {
<           shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
<           shared_load_iterator_.load(aligned_accum_fragment[i]);
<           aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0],
<                                                     aligned_accum_fragment[i]);
<         }
< 
<         shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) *
<                                                  kSmemPointerOffset);
<       }
< 
<       //
<       // Compute the output result
<       //
< 
<       typename OutputTileIterator::Fragment output_fragment;
<       source.apply_output_operator(output_fragment, output_op,
<                                    aligned_accum_fragment[0]);
< 
<       //
<       // Store the final result
<       //
< 
<       destination_iterator.store(output_fragment);
<       ++destination_iterator;
<     }
<   }
< };
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< }  // namespace symmetric
< }  // namespace threadblock
< }  // namespace epilogue
< }  // namespace cutlass
< 
< ////////////////////////////////////////////////////////////////////////////////
``` 
## 19. 文件/symmetric/epilogue/threadblock/predicated_vcol_iterator.h 的修改内容为：
``` 1,33d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< /*! \file
<   \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.
35,414d1
<   The epilogue rearranges the result of a matrix product through shared memory
<   to match canonical tensor layouts in global memory. Epilogues support
<   conversion and reduction operations.
< 
< */
< 
< #pragma once
< 
< #include "cutlass/arch/arch.h"
< #include "cutlass/arch/memory.h"
< #include "cutlass/array.h"
< #include "cutlass/cutlass.h"
< #include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
< #include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
< #include "cutlass/layout/matrix.h"
< #include "cutlass/layout/permute.h"
< #include "cutlass/layout/tensor.h"
< #include "cutlass/matrix_shape.h"
< #include "cutlass/numeric_types.h"
< #include "cutlass/tensor_ref.h"
< #include "cutlass/transform/pitch_linear_thread_map.h"
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace epilogue {
< namespace threadblock {
< namespace symmetric {
< ////////////////////////////////////////////////////////////////////////////////
< 
< /// Tile iterator used to load and store output tile from global memory in
< /// epilogue.
< ///
< /// Satisfies: ReadableTileIterator | PredicatedTileIterator |
< /// ForwardTileIterator
< ///
< template <typename ThreadMap_,    ///< Thread map (conept: OutputTileThreadMap)
<           typename Element_,      ///< Element data type
<           bool ScatterD = false,  ///< Scatter D operand or not
<           typename PermuteDLayout =
<               layout::NoPermute,  ///< Permute D operand or not
<           bool UseCUDAStore = false>
< class PredicatedVColIterator {
<   static_assert(!ScatterD);
<   static_assert(std::is_same<PermuteDLayout, layout::NoPermute>::value);
< 
<  public:
<   using ThreadMap = ThreadMap_;
<   using Shape = typename ThreadMap::Shape;
< 
<   using Element = Element_;
< 
<   using Layout = layout::RowMajor;
<   using TensorRef = TensorRef<Element, Layout>;
<   using ConstTensorRef = typename TensorRef::ConstTensorRef;
< 
<   using Index = typename Layout::Index;
<   using LongIndex = typename Layout::LongIndex;
<   using TensorCoord = MatrixCoord;
< 
<   static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
<   static int const kThreads = ThreadMap::kThreads;
<   static int const kIterations = ThreadMap::Count::kTile;
< 
<   static_assert(ThreadMap::Iterations::kRow > 0,
<                 "ThreadMap::Iterations::kRow must be > 0");
<   static_assert(ThreadMap::Iterations::kGroup > 0,
<                 "ThreadMap::Iterations::kGroup must be > 0");
<   static_assert(ThreadMap::Iterations::kCluster > 0,
<                 "ThreadMap::Iterations::kCluster must be > 0");
<   static_assert(ThreadMap::Iterations::kColumn > 0,
<                 "ThreadMap::Iterations::kColumn must be > 0");
< 
<   /// Fragment object
<   using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
<                                       ThreadMap::Iterations::kRow *
<                                       ThreadMap::Iterations::kGroup *
<                                       ThreadMap::Iterations::kCluster *
<                                       ThreadMap::kElementsPerAccess>;
< 
<   /// Memory access size
<   using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;
< 
<   //
<   // Parameters struct
<   //
< 
<   /// Uses a non-template class
<   struct Params : PredicatedTileIteratorParams {
<     using Base = PredicatedTileIteratorParams;
< 
<     CUTLASS_HOST_DEVICE
<     Params() {}
< 
<     CUTLASS_HOST_DEVICE
<     Params(Layout const &layout)
<         : PredicatedTileIteratorParams(
<               1 * int(sizeof(AccessType)) / kElementsPerAccess,
<               make_OutputTileThreadMapDesc<ThreadMap>()) {}
< 
<     CUTLASS_HOST_DEVICE
<     Params(Base const &base) : Base(base) {}
<   };
< 
<   /// Mask object
<   struct Mask {
<     static int const kCount = ThreadMap::Iterations::kColumn;
< 
<     /// Predicate state
<     bool predicates[kCount];
< 
<     //
<     // Mask
<     //
<     CUTLASS_HOST_DEVICE
<     Mask() { enable(); }
< 
<     ///< Efficiently disables all accesses guarded by mask
<     CUTLASS_HOST_DEVICE void clear() {
<       CUTLASS_PRAGMA_UNROLL
<       for (int i = 0; i < kCount; ++i) {
<         predicates[i] = false;
<       }
<     }
< 
<     ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
<     CUTLASS_DEVICE void enable() {
<       CUTLASS_PRAGMA_UNROLL
<       for (int i = 0; i < kCount; ++i) {
<         predicates[i] = true;
<       }
<     }
<   };
< 
<  private:
<   //
<   // Data members
<   //
< 
<   /// Parameters structure containing reference and precomputed state.
<   PredicatedTileIteratorParams params_;
< 
<   /// Byte-level pointer.
<   uint8_t *byte_pointer_;
< 
<   /// Byte-level pointer for store().
<   uint8_t *store_byte_pointer_;
< 
<   /// Array of boolean values to contain steady-state predicates
<   Mask mask_;
< 
<   /// Extent of the matrix tile in rows
<   Index extent_row_;
< 
<   /// Extent of the matrix tile in rows
<   Index extent_column_;
< 
<   /// A thread's starting row position (assuming steady-state predicates have
<   /// been computed)
<   Index thread_start_row_;
< 
<   /// A thread's starting column
<   Index thread_start_column_;
< 
<   /// Internal state counter
<   int state_[3];
< 
<   //
<   // Static asserts about internal strides
<   //
< 
<   static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
<   static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
<   static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8,
<                 "Expected 64b strides");
< 
<  private:
<   //
<   // Methods
<   //
< 
<  public:
<   //
<   // Methods
<   //
< 
<   /// Constructor
<   CUTLASS_DEVICE
<   PredicatedVColIterator(PredicatedTileIteratorParams const &params,
<                          Element *pointer, TensorCoord extent, int thread_idx,
<                          TensorCoord threadblock_offset = TensorCoord(),
<                          int const *indices = nullptr)
<       : params_(params) {
<     TensorCoord thread_offset =
<         ThreadMap::initial_offset(thread_idx) + threadblock_offset;
< 
<     extent_row_ = extent.row();
<     extent_column_ = extent.column();
< 
<     thread_start_row_ = thread_offset.row();
<     thread_start_column_ = thread_offset.column();
< 
<     // Initialize predicates
<     CUTLASS_PRAGMA_UNROLL
<     for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
<       mask_.predicates[c] = ((thread_offset.column() +
<                               ThreadMap::Delta::kColumn * c) < extent.column());
<     }
< 
<     // Null pointer performs no accesses
<     if (!pointer) {
<       mask_.clear();
<     }
< 
<     // Initialize byte_pointer_
<     byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
<                     LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
<                     LongIndex(thread_offset.column()) * sizeof(AccessType) /
<                         kElementsPerAccess;
< 
<     // store_byte_pointer_ is set to be the same with byte_pointer_
<     store_byte_pointer_ = byte_pointer_;
< 
<     // Initialize internal state counter
<     state_[0] = state_[1] = state_[2] = 0;
< 
<     byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
<                     LongIndex(thread_offset.row()) * LongIndex(params_.stride);
<   }
< 
<   /// Adds a pointer offset in units of Element
<   CUTLASS_HOST_DEVICE
<   void add_pointer_offset(LongIndex pointer_offset) {
<     store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
<     byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
<   }
< 
<   /// Loads a fragment from memory
<   CUTLASS_DEVICE
<   void load_with_byte_offset(Fragment &frag, int64_t byte_offset) const {
<     uint8_t *byte_pointer = byte_pointer_;
<     AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
< 
<     CUTLASS_PRAGMA_UNROLL
<     for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
<          ++cluster) {
<       CUTLASS_PRAGMA_UNROLL
<       for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
<         CUTLASS_PRAGMA_UNROLL
<         for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
<           int frag_row_idx =
<               (row + ThreadMap::Iterations::kRow *
<                          (group + ThreadMap::Iterations::kGroup * cluster));
< 
<           int row_offset = row * ThreadMap::Delta::kRow +
<                            group * ThreadMap::Delta::kGroup +
<                            cluster * ThreadMap::Delta::kCluster;
< 
<           bool row_guard = ((row_offset + thread_start_row_) < extent_row_);
< 
<           CUTLASS_PRAGMA_UNROLL
<           for (int column = 0; column < ThreadMap::Iterations::kColumn;
<                ++column) {
<             bool guard = row_guard && mask_.predicates[column];
<             if (guard) {
<               Element *bias =
<                   reinterpret_cast<Element *>(byte_pointer + byte_offset);
<               frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column]
<                   .fill(*bias);
<             }
<           }
< 
<           if (row + 1 < ThreadMap::Iterations::kRow) {
<             byte_pointer += params_.increment_row;
<           }
<         }
< 
<         if (group + 1 < ThreadMap::Iterations::kGroup) {
<           byte_pointer += params_.increment_group;
<         }
<       }
< 
<       if (cluster + 1 < ThreadMap::Iterations::kCluster) {
<         byte_pointer += params_.increment_cluster;
<       }
<     }
<   }
< 
<   /// Loads a fragment from memory
<   CUTLASS_DEVICE
<   void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }
< 
<   CUTLASS_DEVICE
<   MatrixCoord thread_start() const {
<     return MatrixCoord(thread_start_row_, thread_start_column_);
<   }
< 
<   /// Need to get the thread start row from the tile iterator
<   CUTLASS_DEVICE
<   int32_t thread_start_row() const { return thread_start_row_; }
< 
<   /// Need to get the thread start row from the tile iterator
<   CUTLASS_DEVICE
<   int32_t thread_start_column() const { return thread_start_column_; }
< 
<   /// Extent of the matrix in rows
<   CUTLASS_DEVICE
<   Index extent_row() const { return extent_row_; }
< 
<   /// Extent of the matrix in columns
<   CUTLASS_DEVICE
<   Index extent_column() const { return extent_column_; }
< 
<   /// Advances to the next position to load or store
<   CUTLASS_HOST_DEVICE
<   PredicatedVColIterator &operator++() {
<     ++state_[0];
< 
<     store_byte_pointer_ += params_.advance_row;
< 
<     byte_pointer_ += params_.advance_row;
< 
<     thread_start_row_ += ThreadMap::Shape::kRow;
< 
<     if (state_[0] == ThreadMap::Count::kRow) {
<       state_[0] = 0;
<       ++state_[1];
<       byte_pointer_ += params_.advance_group;
<       store_byte_pointer_ += params_.advance_group;
< 
<       thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
<                            ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
< 
<       if (state_[1] == ThreadMap::Count::kGroup) {
<         state_[1] = 0;
<         ++state_[2];
<         byte_pointer_ += params_.advance_cluster;
<         store_byte_pointer_ += params_.advance_cluster;
< 
<         thread_start_row_ += ThreadMap::Count::kGroup *
<                              ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
<                              ThreadMap::Shape::kRow;
< 
<         if (state_[2] == ThreadMap::Count::kCluster) {
<           state_[2] = 0;
<           byte_pointer_ += params_.advance_tile;
<           store_byte_pointer_ += params_.advance_tile;
< 
<           thread_start_row_ +=
<               ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
<               ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
<         }
<       }
<     }
< 
<     return *this;
<   }
< 
<   ///< Efficiently disables all accesses guarded by mask
<   CUTLASS_DEVICE void clear_mask() { mask_.clear(); }
< 
<   ///< Efficiently enables all accesses guarded by mask
<   CUTLASS_DEVICE void enable_mask() { mask_.enable(); }
< 
<   ///< Sets the mask
<   CUTLASS_DEVICE void get_mask(Mask &mask) const { mask = mask_; }
< 
<   ///< Sets the mask
<   CUTLASS_DEVICE void set_mask(Mask const &mask) { mask_ = mask; }
< };
< 
< }  // namespace symmetric
< }  // namespace threadblock
< }  // namespace epilogue
< }  // namespace cutlass
< 
< ////////////////////////////////////////////////////////////////////////////////
``` 
## 20. 文件/symmetric/epilogue/thread/linear_combination_dequant.h 的修改内容为：
``` 1,36d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< /*! \file
<   \brief Functor performing linear combination operations used by dequantize
<   epilogues.
< */
< #pragma once
38,278d1
< 
< 
< #include "cutlass/array.h"
< #include "cutlass/cutlass.h"
< #include "cutlass/epilogue/thread/linear_combination_params.h"
< #include "cutlass/epilogue/thread/scale_type.h"
< #include "cutlass/functional.h"
< #include "cutlass/numeric_conversion.h"
< #include "cutlass/numeric_types.h"
< #include<cuda_fp16.h>
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< namespace epilogue {
< namespace thread {
< namespace symmetric {
< 
< struct MyScaleType {
<   enum Kind {
<     Dequantize,
<   };
< };
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< template <typename ElementOutput_, int Count, typename ElementAccumulator_,
<           typename ElementCompute_ = cutlass::half_t,
<           MyScaleType::Kind Scale = MyScaleType::Dequantize,
<           FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
<           typename ElementSource_ = cutlass::half_t>
< class LinearCombinationDequant {
<  public:
<   using ElementOutput = ElementOutput_;
<   using ElementSource = ElementSource_;
<   using ElementAccumulator = ElementAccumulator_;
<   using ElementCompute = ElementCompute_;
<   using ElementC = ElementSource_;
<   using ElementD = ElementOutput_;
< 
<   static int const kCount = Count;
<   static const MyScaleType::Kind kScale = MyScaleType::Dequantize;
< 
<   using FragmentOutput = Array<ElementOutput, kCount>;
<   using FragmentSource = Array<ElementSource, kCount>;
<   using FragmentAccumulator = Array<ElementAccumulator, kCount>;
<   using FragmentCompute = Array<ElementCompute, kCount>;
< 
<   static FloatRoundStyle const kRound = Round;
< 
<   struct Params {
<     ElementCompute beta;
< 
<     CUTLASS_HOST_DEVICE
<     Params() : beta(ElementCompute(0)) {}
< 
<     CUTLASS_HOST_DEVICE
<     Params(ElementCompute beta) : beta(beta) {}
<   };
< 
<  private:
<   //
<   // Data members
<   //
< 
<   ElementCompute beta_ = ElementCompute(0);
< 
<  public:
<   /// Constructs the function object
<   CUTLASS_HOST_DEVICE
<   LinearCombinationDequant(Params const &params) { beta_ = params.beta; }
< 
<   /// Returns true if source is needed
<   CUTLASS_HOST_DEVICE
<   bool is_source_needed() const { return true; }
< 
<   CUTLASS_HOST_DEVICE
<   void set_k_partition(int k_partition, int k_partition_count) {
<     if (k_partition) {
<       beta_ = ElementCompute(1);
<     }
<   }
< 
<   CUTLASS_HOST_DEVICE
<   FragmentOutput operator()(FragmentAccumulator const &accumulator,
<                             FragmentSource const &source,
<                             FragmentSource const &row_vec_alpha,
<                             FragmentSource const &col_vec_alpha) const {
<     NumericArrayConverter<ElementCompute, ElementSource, kCount, Round>
<         source_converter;
<     NumericArrayConverter<int32_t, ElementAccumulator, kCount, Round>
<         accumulator_converter;
< 
<     NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
<         destination_converter;
< 
<     FragmentCompute converted_source = source_converter(source);
<     FragmentCompute converted_row_vec_alpha = source_converter(row_vec_alpha);
<     FragmentCompute converted_col_vec_alpha = source_converter(col_vec_alpha);
< 
<     using FragmentComputeFloat = Array<int32_t, kCount >;
<     FragmentComputeFloat converted_accumulator = accumulator_converter(accumulator);
< 
<     FragmentCompute result;
<     half_t *result_ptr = reinterpret_cast<half_t *>(&result);
<     const half_t *source_ptr =
<         reinterpret_cast<const half_t *>(&converted_source);
<     const int32_t *acc_ptr =
<         reinterpret_cast<const int32_t *>(&converted_accumulator);
<     const half_t *row_vec_ptr =
<         reinterpret_cast<const half_t *>(&converted_row_vec_alpha);
<     const half_t *col_vec_ptr =
<         reinterpret_cast<const half_t *>(&converted_col_vec_alpha);
< 
<     
<     CUTLASS_PRAGMA_UNROLL
<     for (int i = 0; i < kCount; ++i) {
<       result_ptr[i] =
<           (__float2half)(  (  static_cast<float>(acc_ptr[i]) ) * 
<           (static_cast<float>(col_vec_ptr[i]) * static_cast<float>(row_vec_ptr[i]))
<            +   static_cast<float>(source_ptr[i]));
<     }
< 
<     return destination_converter(result);
<   }
< };
< 
< 
< 
< 
< /////////////////////////////////////////////////////////////////////////////////////////////////
< __forceinline__ __host__ __device__ float silu(float x)
< {
<     return x / (1.f + expf(-x));
< }
< template <typename ElementOutput_, int Count, typename ElementAccumulator_,
<           typename ElementCompute_ = cutlass::half_t,
<           MyScaleType::Kind Scale = MyScaleType::Dequantize,
<           FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
<           typename ElementSource_ = cutlass::half_t>
< class LinearCombinationDequantSilu {
<  public:
<   using ElementOutput = ElementOutput_;
<   using ElementSource = ElementSource_;
<   using ElementAccumulator = ElementAccumulator_;
<   using ElementCompute = ElementCompute_;
<   using ElementC = ElementSource_;
<   using ElementD = ElementOutput_;
< 
<   static int const kCount = Count;
<   static const MyScaleType::Kind kScale = MyScaleType::Dequantize;
< 
<   using FragmentOutput = Array<ElementOutput, kCount>;
<   using FragmentSource = Array<ElementSource, kCount>;
<   using FragmentAccumulator = Array<ElementAccumulator, kCount>;
<   using FragmentCompute = Array<ElementCompute, kCount>;
< 
<   static FloatRoundStyle const kRound = Round;
< 
<   struct Params {
<     ElementCompute beta;
< 
<     CUTLASS_HOST_DEVICE
<     Params() : beta(ElementCompute(0)) {}
< 
<     CUTLASS_HOST_DEVICE
<     Params(ElementCompute beta) : beta(beta) {}
<   };
< 
<  private:
<   //
<   // Data members
<   //
< 
<   ElementCompute beta_ = ElementCompute(0);
< 
<  public:
<   /// Constructs the function object
<   CUTLASS_HOST_DEVICE
<   LinearCombinationDequantSilu(Params const &params) { beta_ = params.beta; }
< 
<   /// Returns true if source is needed
<   CUTLASS_HOST_DEVICE
<   bool is_source_needed() const { return true; }
< 
<   CUTLASS_HOST_DEVICE
<   void set_k_partition(int k_partition, int k_partition_count) {
<     if (k_partition) {
<       beta_ = ElementCompute(1);
<     }
<   }
< 
<   CUTLASS_HOST_DEVICE
<   FragmentOutput operator()(FragmentAccumulator const &accumulator,
<                             FragmentSource const &source,
<                             FragmentSource const &row_vec_alpha,
<                             FragmentSource const &col_vec_alpha) const {
<     NumericArrayConverter<ElementCompute, ElementSource, kCount, Round>
<         source_converter;
<     NumericArrayConverter<int32_t, ElementAccumulator, kCount, Round>
<         accumulator_converter;
< 
<     NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
<         destination_converter;
< 
<     FragmentCompute converted_source = source_converter(source);
<     FragmentCompute converted_row_vec_alpha = source_converter(row_vec_alpha);
<     FragmentCompute converted_col_vec_alpha = source_converter(col_vec_alpha);
< 
<     using FragmentComputeFloat = Array<int32_t, kCount >;
<     FragmentComputeFloat converted_accumulator = accumulator_converter(accumulator);
< 
<     FragmentCompute result;
<     half_t *result_ptr = reinterpret_cast<half_t *>(&result);
<     const half_t *source_ptr =
<         reinterpret_cast<const half_t *>(&converted_source);
<     const int32_t *acc_ptr =
<         reinterpret_cast<const int32_t *>(&converted_accumulator);
<     const half_t *row_vec_ptr =
<         reinterpret_cast<const half_t *>(&converted_row_vec_alpha);
<     const half_t *col_vec_ptr =
<         reinterpret_cast<const half_t *>(&converted_col_vec_alpha);
< 
<     
<     CUTLASS_PRAGMA_UNROLL
<     for (int i = 0; i < kCount; ++i) {
<       float tmp =
<           silu(  (  static_cast<float>(acc_ptr[i]) ) * (static_cast<float>(col_vec_ptr[i]) * static_cast<float>(row_vec_ptr[i]))
<            +   static_cast<float>(source_ptr[i]));
<  
<       result_ptr[i] =
<           (__float2half)(tmp);
<     }
< 
<     return destination_converter(result);
<   }
< };
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< }  // namespace symmetric
< }  // namespace thread
< }  // namespace epilogue
< }  // namespace cutlass
``` 
## 21. 文件/symmetric/gemm/device/gemm_dequantsilu.h 的修改内容为：
``` 1,35d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< /*! \file
<     \brief Template for a pipelined GEMM kernel. Does not compute batching or
<    support split-K.
< */
37,382d1
< #pragma once
< 
< #include "cutlass/arch/arch.h"
< #include "cutlass/cutlass.h"
< #include "cutlass/device_kernel.h"
< #include "cutlass/gemm/device/default_gemm_configuration.h"
< #include "cutlass/gemm/kernel/gemm.h"
< #include "cutlass/gemm/threadblock/threadblock_swizzle.h"
< #include "cutlass/layout/permute.h"
< #include "cutlass/numeric_types.h"
< #include "symmetric/epilogue/thread/linear_combination_dequant.h"
< #include "symmetric/gemm/kernel/default_gemm_dequant.h"
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< namespace gemm {
< namespace device {
< namespace symmetric {
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< template <
<     /// Element type for A matrix operand
<     typename ElementA_,
<     /// Layout type for A matrix operand
<     typename LayoutA_,
<     /// Element type for B matrix operand
<     typename ElementB_,
<     /// Layout type for B matrix operand
<     typename LayoutB_,
<     /// Element type for C and D matrix operands
<     typename ElementC_,
<     /// Layout type for C and D matrix operands
<     typename LayoutC_,
<     /// Element type for internal accumulation
<     typename ElementAccumulator_ = ElementC_,
<     /// Operator class tag
<     typename OperatorClass_ = arch::OpClassTensorOp,
<     /// Tag indicating architecture to tune for
<     typename ArchTag_ = arch::Sm80,
<     /// Threadblock-level tile size (concept: GemmShape)
<     typename ThreadblockShape_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::ThreadblockShape,
<     /// Warp-level tile size (concept: GemmShape)
<     typename WarpShape_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::WarpShape,
<     /// Instruction-level tile size (concept: GemmShape)
<     typename InstructionShape_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::InstructionShape,
<     /// Epilogue output operator
<     typename EpilogueOutputOp_ =
<         cutlass::epilogue::thread::symmetric::LinearCombinationDequantSilu<
<             ElementC_, 128 / cutlass::sizeof_bits<ElementC_>::value,
<             ElementAccumulator_, ElementC_,
<             cutlass::epilogue::thread::symmetric::MyScaleType::Dequantize,
<             cutlass::FloatRoundStyle::round_to_nearest, ElementC_>,
<     /// Threadblock-level swizzling operator
<     typename ThreadblockSwizzle_ =
<         typename threadblock::GemmIdentityThreadblockSwizzle<>,
<     /// Number of stages used in the pipelined mainloop
<     int Stages =
<         DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
<                                  ElementC_, ElementAccumulator_>::kStages,
<     /// Access granularity of A matrix in units of elements
<     int AlignmentA =
<         DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
<                                  ElementC_, ElementAccumulator_>::kAlignmentA,
<     /// Access granularity of B matrix in units of elements
<     int AlignmentB =
<         DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
<                                  ElementC_, ElementAccumulator_>::kAlignmentB,
<     /// If true, kernel supports split-K with serial reduction
<     bool SplitKSerial = false,
<     /// Operation performed by GEMM
<     typename Operator_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::Operator,
<     /// Gather operand A by using an index array
<     bool GatherA = false,
<     /// Gather operand B by using an index array
<     bool GatherB = false,
<     /// Scatter result D by using an index array
<     bool ScatterD = false,
<     /// Permute result D
<     typename PermuteDLayout = layout::NoPermute>
< class GemmDequantSilu {
<  public:
<   using ElementA = ElementA_;
<   using LayoutA = LayoutA_;
<   using TensorRefA = TensorRef<ElementA const, LayoutA>;
<   using ElementB = ElementB_;
<   using LayoutB = LayoutB_;
<   using TensorRefB = TensorRef<ElementB const, LayoutB>;
<   using ElementC = ElementC_;
<   using LayoutC = LayoutC_;
<   using TensorRefC = TensorRef<ElementC const, LayoutC>;
<   using TensorRefD = TensorRef<ElementC, LayoutC>;
<   using ElementAccumulator = ElementAccumulator_;
<   using OperatorClass = OperatorClass_;
<   using ArchTag = ArchTag_;
<   using ThreadblockShape = ThreadblockShape_;
<   using WarpShape = WarpShape_;
<   using InstructionShape = InstructionShape_;
<   using EpilogueOutputOp = EpilogueOutputOp_;
<   using ThreadblockSwizzle = ThreadblockSwizzle_;
<   using Operator = Operator_;
<   static int const kStages = Stages;
<   static int const kAlignmentA = AlignmentA;
<   static int const kAlignmentB = AlignmentB;
<   static int const kAlignmentC = EpilogueOutputOp::kCount;
<   static bool const kSplitKSerial = SplitKSerial;
<   static ComplexTransform const kTransformA = ComplexTransform::kNone;
<   static ComplexTransform const kTransformB = ComplexTransform::kNone;
< 
<   /// Define the kernel
<   using GemmKernel = typename kernel::symmetric::DefaultGemmDequant<
<       ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
<       LayoutC, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
<       WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
<       kStages, kSplitKSerial, Operator, SharedMemoryClearOption::kNone, GatherA,
<       GatherB, ScatterD, PermuteDLayout>::GemmKernel;
< 
<   /// Argument structure
<   struct Arguments {
<     //
<     // Data members
<     //
< 
<     GemmCoord problem_size;
<     TensorRef<ElementA const, LayoutA> ref_A;
<     TensorRef<ElementB const, LayoutB> ref_B;
<     TensorRef<ElementC const, LayoutC> ref_C;
<     TensorRef<ElementC, LayoutC> ref_D;
<     TensorRef<ElementC const, LayoutC> ref_row_vec;
<     TensorRef<ElementC const, LayoutC> ref_col_vec;
<     typename EpilogueOutputOp::Params epilogue;
<     int split_k_slices;
<     // For gather+scatter operations
<     int const *gather_A_indices;
<     int const *gather_B_indices;
<     int const *scatter_D_indices;
< 
<     //
<     // Methods
<     //
< 
<     /// Default ctor
<     CUTLASS_HOST_DEVICE
<     Arguments() : problem_size(0, 0, 0), split_k_slices(1) {}
< 
<     /// Constructs an Arguments structure
<     CUTLASS_HOST_DEVICE
<     Arguments(GemmCoord problem_size_,
<               TensorRef<ElementA const, LayoutA> ref_A_,
<               TensorRef<ElementB const, LayoutB> ref_B_,
<               TensorRef<ElementC const, LayoutC> ref_C_,
<               TensorRef<ElementC, LayoutC> ref_D_,
<               TensorRef<ElementC const, LayoutC> ref_row_vec_,
<               TensorRef<ElementC const, LayoutC> ref_col_vec_,
<               typename EpilogueOutputOp::Params epilogue_ =
<                   typename EpilogueOutputOp::Params(),
<               int split_k_slices = 1, int const *gather_A_indices_ = nullptr,
<               int const *gather_B_indices_ = nullptr,
<               int const *scatter_D_indices_ = nullptr)
<         : problem_size(problem_size_),
<           ref_A(ref_A_),
<           ref_B(ref_B_),
<           ref_C(ref_C_),
<           ref_D(ref_D_),
<           ref_row_vec(ref_row_vec_),
<           ref_col_vec(ref_col_vec_),
<           epilogue(epilogue_),
<           split_k_slices(split_k_slices),
<           gather_A_indices(gather_A_indices_),
<           gather_B_indices(gather_B_indices_),
<           scatter_D_indices(scatter_D_indices_) {}
<   };
< 
<  private:
<   /// Kernel parameters object
<   typename GemmKernel::Params params_;
< 
<  public:
<   /// Constructs the GEMM.
<   GemmDequantSilu() {}
< 
<   /// Determines whether the GEMM can execute the given problem.
<   static Status can_implement(Arguments const &args) {
<     if (!kSplitKSerial && args.split_k_slices > 1) {
<       return Status::kErrorInvalidProblem;
<     }
< 
<     Status status = GemmKernel::can_implement(
<         args.problem_size, args.ref_A.non_const_ref(),
<         args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D,
<         args.ref_row_vec.non_const_ref(), args.ref_col_vec.non_const_ref());
< 
<     if (status != Status::kSuccess) {
<       return status;
<     }
< 
<     return Status::kSuccess;
<   }
< 
<   /// Gets the workspace size
<   static size_t get_workspace_size(Arguments const &args) {
<     size_t bytes = 0;
< 
<     // Determine grid shape
<     ThreadblockSwizzle threadblock_swizzle;
< 
<     cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
<         args.problem_size,
<         {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
<         args.split_k_slices);
< 
<     if (kSplitKSerial && args.split_k_slices > 1) {
<       bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
<     }
< 
<     return bytes;
<   }
< 
<   /// Initializes GEMM state from arguments.
<   Status initialize(Arguments const &args, void *workspace = nullptr,
<                     cudaStream_t stream = nullptr) {
<     // Determine grid shape
<     ThreadblockSwizzle threadblock_swizzle;
< 
<     cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
<         args.problem_size,
<         {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
<         args.split_k_slices);
< 
<     if (kSplitKSerial) {
<       if (args.split_k_slices > 1) {
<         if (!workspace) {
<           return Status::kErrorWorkspaceNull;
<         }
< 
<         size_t bytes = get_workspace_size(args);
< 
<         cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);
< 
<         if (result != cudaSuccess) {
<           return Status::kErrorInternal;
<         }
<       }
<     } else {
<       if (args.split_k_slices > 1) {
<         return Status::kErrorInvalidProblem;
<       }
<     }
< 
<     // Initialize the Params structure
<     params_ = typename GemmKernel::Params{args.problem_size,
<                                           grid_shape,
<                                           args.ref_A.non_const_ref(),
<                                           args.ref_B.non_const_ref(),
<                                           args.ref_C.non_const_ref(),
<                                           args.ref_D,
<                                           args.ref_row_vec.non_const_ref(),
<                                           args.ref_col_vec.non_const_ref(),
<                                           args.epilogue,
<                                           static_cast<int *>(workspace),
<                                           args.gather_A_indices,
<                                           args.gather_B_indices,
<                                           args.scatter_D_indices};
< 
<     return Status::kSuccess;
<   }
< 
<   /// Lightweight update given a subset of arguments
<   Status update(Arguments const &args, void *workspace = nullptr) {
<     if (kSplitKSerial && args.split_k_slices > 1) {
<       if (!workspace) {
<         return Status::kErrorWorkspaceNull;
<       }
<     }
< 
<     params_.ref_A.reset(args.ref_A.non_const_ref().data());
<     params_.ref_B.reset(args.ref_B.non_const_ref().data());
<     params_.ref_C.reset(args.ref_C.non_const_ref().data());
<     params_.ref_D.reset(args.ref_D.data());
<     params_.ref_row_vec.reset(args.ref_row_vec.non_const_ref().data());
<     params_.ref_col_vec.reset(args.ref_col_vec.non_const_ref().data());
<     params_.output_op = args.epilogue;
<     params_.semaphore = static_cast<int *>(workspace);
< 
<     return Status::kSuccess;
<   }
< 
<   /// Runs the kernel using initialized state.
<   Status run(cudaStream_t stream = nullptr) {
<     ThreadblockSwizzle threadblock_swizzle;
< 
<     dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
<     dim3 block(GemmKernel::kThreadCount, 1, 1);
< 
<     cudaError_t result;
< 
<     int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
< 
<     if (smem_size >= (48 << 10)) {
<       result = cudaFuncSetAttribute(Kernel<GemmKernel>,
<                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
<                                     smem_size);
< 
<       if (result != cudaSuccess) {
<         return Status::kErrorInternal;
<       }
<     }
< 
<     cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
< 
<     result = cudaGetLastError();
< 
<     return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
<   }
< 
<   /// Runs the kernel using initialized state.
<   Status operator()(cudaStream_t stream = nullptr) { return run(stream); }
< 
<   /// Runs the kernel using initialized state.
<   Status operator()(Arguments const &args, void *workspace = nullptr,
<                     cudaStream_t stream = nullptr) {
<     Status status = initialize(args, workspace, stream);
< 
<     if (status == Status::kSuccess) {
<       status = run(stream);
<     }
< 
<     return status;
<   }
< };
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< }  // namespace symmetric
< }  // namespace device
< }  // namespace gemm
< }  // namespace cutlass
< 
< ////////////////////////////////////////////////////////////////////////////////
``` 
## 22. 文件/symmetric/gemm/device/gemm_dequant.h 的修改内容为：
``` 1,35d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< /*! \file
<     \brief Template for a pipelined GEMM kernel. Does not compute batching or
<    support split-K.
< */
37,382d1
< #pragma once
< 
< #include "cutlass/arch/arch.h"
< #include "cutlass/cutlass.h"
< #include "cutlass/device_kernel.h"
< #include "cutlass/gemm/device/default_gemm_configuration.h"
< #include "cutlass/gemm/kernel/gemm.h"
< #include "cutlass/gemm/threadblock/threadblock_swizzle.h"
< #include "cutlass/layout/permute.h"
< #include "cutlass/numeric_types.h"
< #include "symmetric/epilogue/thread/linear_combination_dequant.h"
< #include "symmetric/gemm/kernel/default_gemm_dequant.h"
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< namespace gemm {
< namespace device {
< namespace symmetric {
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< template <
<     /// Element type for A matrix operand
<     typename ElementA_,
<     /// Layout type for A matrix operand
<     typename LayoutA_,
<     /// Element type for B matrix operand
<     typename ElementB_,
<     /// Layout type for B matrix operand
<     typename LayoutB_,
<     /// Element type for C and D matrix operands
<     typename ElementC_,
<     /// Layout type for C and D matrix operands
<     typename LayoutC_,
<     /// Element type for internal accumulation
<     typename ElementAccumulator_ = ElementC_,
<     /// Operator class tag
<     typename OperatorClass_ = arch::OpClassTensorOp,
<     /// Tag indicating architecture to tune for
<     typename ArchTag_ = arch::Sm80,
<     /// Threadblock-level tile size (concept: GemmShape)
<     typename ThreadblockShape_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::ThreadblockShape,
<     /// Warp-level tile size (concept: GemmShape)
<     typename WarpShape_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::WarpShape,
<     /// Instruction-level tile size (concept: GemmShape)
<     typename InstructionShape_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::InstructionShape,
<     /// Epilogue output operator
<     typename EpilogueOutputOp_ =
<         cutlass::epilogue::thread::symmetric::LinearCombinationDequant<
<             ElementC_, 128 / cutlass::sizeof_bits<ElementC_>::value,
<             ElementAccumulator_, ElementC_,
<             cutlass::epilogue::thread::symmetric::MyScaleType::Dequantize,
<             cutlass::FloatRoundStyle::round_to_nearest, ElementC_>,
<     /// Threadblock-level swizzling operator
<     typename ThreadblockSwizzle_ =
<         typename threadblock::GemmIdentityThreadblockSwizzle<>,
<     /// Number of stages used in the pipelined mainloop
<     int Stages =
<         DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
<                                  ElementC_, ElementAccumulator_>::kStages,
<     /// Access granularity of A matrix in units of elements
<     int AlignmentA =
<         DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
<                                  ElementC_, ElementAccumulator_>::kAlignmentA,
<     /// Access granularity of B matrix in units of elements
<     int AlignmentB =
<         DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
<                                  ElementC_, ElementAccumulator_>::kAlignmentB,
<     /// If true, kernel supports split-K with serial reduction
<     bool SplitKSerial = false,
<     /// Operation performed by GEMM
<     typename Operator_ = typename DefaultGemmConfiguration<
<         OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
<         ElementAccumulator_>::Operator,
<     /// Gather operand A by using an index array
<     bool GatherA = false,
<     /// Gather operand B by using an index array
<     bool GatherB = false,
<     /// Scatter result D by using an index array
<     bool ScatterD = false,
<     /// Permute result D
<     typename PermuteDLayout = layout::NoPermute>
< class GemmDequant {
<  public:
<   using ElementA = ElementA_;
<   using LayoutA = LayoutA_;
<   using TensorRefA = TensorRef<ElementA const, LayoutA>;
<   using ElementB = ElementB_;
<   using LayoutB = LayoutB_;
<   using TensorRefB = TensorRef<ElementB const, LayoutB>;
<   using ElementC = ElementC_;
<   using LayoutC = LayoutC_;
<   using TensorRefC = TensorRef<ElementC const, LayoutC>;
<   using TensorRefD = TensorRef<ElementC, LayoutC>;
<   using ElementAccumulator = ElementAccumulator_;
<   using OperatorClass = OperatorClass_;
<   using ArchTag = ArchTag_;
<   using ThreadblockShape = ThreadblockShape_;
<   using WarpShape = WarpShape_;
<   using InstructionShape = InstructionShape_;
<   using EpilogueOutputOp = EpilogueOutputOp_;
<   using ThreadblockSwizzle = ThreadblockSwizzle_;
<   using Operator = Operator_;
<   static int const kStages = Stages;
<   static int const kAlignmentA = AlignmentA;
<   static int const kAlignmentB = AlignmentB;
<   static int const kAlignmentC = EpilogueOutputOp::kCount;
<   static bool const kSplitKSerial = SplitKSerial;
<   static ComplexTransform const kTransformA = ComplexTransform::kNone;
<   static ComplexTransform const kTransformB = ComplexTransform::kNone;
< 
<   /// Define the kernel
<   using GemmKernel = typename kernel::symmetric::DefaultGemmDequant<
<       ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
<       LayoutC, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
<       WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
<       kStages, kSplitKSerial, Operator, SharedMemoryClearOption::kNone, GatherA,
<       GatherB, ScatterD, PermuteDLayout>::GemmKernel;
< 
<   /// Argument structure
<   struct Arguments {
<     //
<     // Data members
<     //
< 
<     GemmCoord problem_size;
<     TensorRef<ElementA const, LayoutA> ref_A;
<     TensorRef<ElementB const, LayoutB> ref_B;
<     TensorRef<ElementC const, LayoutC> ref_C;
<     TensorRef<ElementC, LayoutC> ref_D;
<     TensorRef<ElementC const, LayoutC> ref_row_vec;
<     TensorRef<ElementC const, LayoutC> ref_col_vec;
<     typename EpilogueOutputOp::Params epilogue;
<     int split_k_slices;
<     // For gather+scatter operations
<     int const *gather_A_indices;
<     int const *gather_B_indices;
<     int const *scatter_D_indices;
< 
<     //
<     // Methods
<     //
< 
<     /// Default ctor
<     CUTLASS_HOST_DEVICE
<     Arguments() : problem_size(0, 0, 0), split_k_slices(1) {}
< 
<     /// Constructs an Arguments structure
<     CUTLASS_HOST_DEVICE
<     Arguments(GemmCoord problem_size_,
<               TensorRef<ElementA const, LayoutA> ref_A_,
<               TensorRef<ElementB const, LayoutB> ref_B_,
<               TensorRef<ElementC const, LayoutC> ref_C_,
<               TensorRef<ElementC, LayoutC> ref_D_,
<               TensorRef<ElementC const, LayoutC> ref_row_vec_,
<               TensorRef<ElementC const, LayoutC> ref_col_vec_,
<               typename EpilogueOutputOp::Params epilogue_ =
<                   typename EpilogueOutputOp::Params(),
<               int split_k_slices = 1, int const *gather_A_indices_ = nullptr,
<               int const *gather_B_indices_ = nullptr,
<               int const *scatter_D_indices_ = nullptr)
<         : problem_size(problem_size_),
<           ref_A(ref_A_),
<           ref_B(ref_B_),
<           ref_C(ref_C_),
<           ref_D(ref_D_),
<           ref_row_vec(ref_row_vec_),
<           ref_col_vec(ref_col_vec_),
<           epilogue(epilogue_),
<           split_k_slices(split_k_slices),
<           gather_A_indices(gather_A_indices_),
<           gather_B_indices(gather_B_indices_),
<           scatter_D_indices(scatter_D_indices_) {}
<   };
< 
<  private:
<   /// Kernel parameters object
<   typename GemmKernel::Params params_;
< 
<  public:
<   /// Constructs the GEMM.
<   GemmDequant() {}
< 
<   /// Determines whether the GEMM can execute the given problem.
<   static Status can_implement(Arguments const &args) {
<     if (!kSplitKSerial && args.split_k_slices > 1) {
<       return Status::kErrorInvalidProblem;
<     }
< 
<     Status status = GemmKernel::can_implement(
<         args.problem_size, args.ref_A.non_const_ref(),
<         args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D,
<         args.ref_row_vec.non_const_ref(), args.ref_col_vec.non_const_ref());
< 
<     if (status != Status::kSuccess) {
<       return status;
<     }
< 
<     return Status::kSuccess;
<   }
< 
<   /// Gets the workspace size
<   static size_t get_workspace_size(Arguments const &args) {
<     size_t bytes = 0;
< 
<     // Determine grid shape
<     ThreadblockSwizzle threadblock_swizzle;
< 
<     cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
<         args.problem_size,
<         {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
<         args.split_k_slices);
< 
<     if (kSplitKSerial && args.split_k_slices > 1) {
<       bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
<     }
< 
<     return bytes;
<   }
< 
<   /// Initializes GEMM state from arguments.
<   Status initialize(Arguments const &args, void *workspace = nullptr,
<                     cudaStream_t stream = nullptr) {
<     // Determine grid shape
<     ThreadblockSwizzle threadblock_swizzle;
< 
<     cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
<         args.problem_size,
<         {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
<         args.split_k_slices);
< 
<     if (kSplitKSerial) {
<       if (args.split_k_slices > 1) {
<         if (!workspace) {
<           return Status::kErrorWorkspaceNull;
<         }
< 
<         size_t bytes = get_workspace_size(args);
< 
<         cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);
< 
<         if (result != cudaSuccess) {
<           return Status::kErrorInternal;
<         }
<       }
<     } else {
<       if (args.split_k_slices > 1) {
<         return Status::kErrorInvalidProblem;
<       }
<     }
< 
<     // Initialize the Params structure
<     params_ = typename GemmKernel::Params{args.problem_size,
<                                           grid_shape,
<                                           args.ref_A.non_const_ref(),
<                                           args.ref_B.non_const_ref(),
<                                           args.ref_C.non_const_ref(),
<                                           args.ref_D,
<                                           args.ref_row_vec.non_const_ref(),
<                                           args.ref_col_vec.non_const_ref(),
<                                           args.epilogue,
<                                           static_cast<int *>(workspace),
<                                           args.gather_A_indices,
<                                           args.gather_B_indices,
<                                           args.scatter_D_indices};
< 
<     return Status::kSuccess;
<   }
< 
<   /// Lightweight update given a subset of arguments
<   Status update(Arguments const &args, void *workspace = nullptr) {
<     if (kSplitKSerial && args.split_k_slices > 1) {
<       if (!workspace) {
<         return Status::kErrorWorkspaceNull;
<       }
<     }
< 
<     params_.ref_A.reset(args.ref_A.non_const_ref().data());
<     params_.ref_B.reset(args.ref_B.non_const_ref().data());
<     params_.ref_C.reset(args.ref_C.non_const_ref().data());
<     params_.ref_D.reset(args.ref_D.data());
<     params_.ref_row_vec.reset(args.ref_row_vec.non_const_ref().data());
<     params_.ref_col_vec.reset(args.ref_col_vec.non_const_ref().data());
<     params_.output_op = args.epilogue;
<     params_.semaphore = static_cast<int *>(workspace);
< 
<     return Status::kSuccess;
<   }
< 
<   /// Runs the kernel using initialized state.
<   Status run(cudaStream_t stream = nullptr) {
<     ThreadblockSwizzle threadblock_swizzle;
< 
<     dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
<     dim3 block(GemmKernel::kThreadCount, 1, 1);
< 
<     cudaError_t result;
< 
<     int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
< 
<     if (smem_size >= (48 << 10)) {
<       result = cudaFuncSetAttribute(Kernel<GemmKernel>,
<                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
<                                     smem_size);
< 
<       if (result != cudaSuccess) {
<         return Status::kErrorInternal;
<       }
<     }
< 
<     cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
< 
<     result = cudaGetLastError();
< 
<     return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
<   }
< 
<   /// Runs the kernel using initialized state.
<   Status operator()(cudaStream_t stream = nullptr) { return run(stream); }
< 
<   /// Runs the kernel using initialized state.
<   Status operator()(Arguments const &args, void *workspace = nullptr,
<                     cudaStream_t stream = nullptr) {
<     Status status = initialize(args, workspace, stream);
< 
<     if (status == Status::kSuccess) {
<       status = run(stream);
<     }
< 
<     return status;
<   }
< };
< 
< ////////////////////////////////////////////////////////////////////////////////
< 
< }  // namespace symmetric
< }  // namespace device
< }  // namespace gemm
< }  // namespace cutlass
< 
< ////////////////////////////////////////////////////////////////////////////////
``` 
## 23. 文件/symmetric/gemm/kernel/default_gemm_dequant.h 的修改内容为：
``` 1,32d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
< #pragma once
34,136d1
< #include "cutlass/gemm/kernel/default_gemm.h"
< #include "symmetric/epilogue/threadblock/default_epilogue_tensor_op_dequant.h"
< #include "symmetric/gemm/kernel/gemm_dequant.h"
< ////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< namespace gemm {
< namespace kernel {
< namespace symmetric {
< 
< template <
<     /// Element type for A matrix operand
<     typename ElementA_,
<     /// Layout type for A matrix operand
<     typename LayoutA_,
<     /// Access granularity of A matrix in units of elements
<     int kAlignmentA,
<     /// Element type for B matrix operand
<     typename ElementB_,
<     /// Layout type for B matrix operand
<     typename LayoutB_,
<     /// Access granularity of B matrix in units of elements
<     int kAlignmentB,
<     /// Element type for C and D matrix operands
<     typename ElementC_,
<     /// Layout type for C and D matrix operands
<     typename LayoutC_,
<     /// Element type for internal accumulation
<     typename ElementAccumulator,
<     /// Operator class tag
<     typename OperatorClass,
<     /// Tag indicating architecture to tune for
<     typename ArchTag,
<     /// Threadblock-level tile size (concept: GemmShape)
<     typename ThreadblockShape,
<     /// Warp-level tile size (concept: GemmShape)
<     typename WarpShape,
<     /// Warp-level tile size (concept: GemmShape)
<     typename InstructionShape,
<     /// Epilogue output operator
<     typename EpilogueOutputOp,
<     /// Threadblock-level swizzling operator
<     typename ThreadblockSwizzle,
<     /// Number of stages used in the pipelined mainloop
<     int Stages,
<     /// If true, kernel is configured to support serial reduction in the
<     /// epilogue
<     bool SplitKSerial,
<     /// Operation performed by GEMM
<     typename Operator,
<     /// Use zfill or predicate for out-of-bound cp.async
<     SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
<     /// Gather operand A by using an index array
<     bool GatherA = false,
<     /// Gather operand B by using an index array
<     bool GatherB = false,
<     /// Scatter result D by using an index array
<     bool ScatterD = false,
<     /// Permute result D
<     typename PermuteDLayout = layout::NoPermute,
<     /// Permute operand A
<     typename PermuteALayout = layout::NoPermute,
<     /// Permute operand B
<     typename PermuteBLayout = layout::NoPermute,
<     ///
<     typename Enable = void>
< struct DefaultGemmDequant
<     : public DefaultGemm<ElementA_, LayoutA_, kAlignmentA, ElementB_, LayoutB_,
<                          kAlignmentB, ElementC_, LayoutC_, ElementAccumulator,
<                          arch::OpClassTensorOp, arch::Sm90, ThreadblockShape,
<                          WarpShape, InstructionShape, EpilogueOutputOp,
<                          ThreadblockSwizzle, Stages, SplitKSerial, Operator,
<                          SharedMemoryClear, GatherA, GatherB, ScatterD,
<                          PermuteDLayout, PermuteALayout, PermuteBLayout> {
<   static_assert((platform::is_same<LayoutC_, layout::RowMajor>::value ||
<                  platform::is_same<LayoutC_, layout::AffineRankN<2>>::value),
<                 "Epilogue in the kernel level must be row major");
< 
<   using DefaultGemm =
<       DefaultGemm<ElementA_, LayoutA_, kAlignmentA, ElementB_, LayoutB_,
<                   kAlignmentB, ElementC_, LayoutC_, ElementAccumulator,
<                   arch::OpClassTensorOp, arch::Sm90, ThreadblockShape,
<                   WarpShape, InstructionShape, EpilogueOutputOp,
<                   ThreadblockSwizzle, Stages, SplitKSerial, Operator,
<                   SharedMemoryClear, GatherA, GatherB, ScatterD, PermuteDLayout,
<                   PermuteALayout, PermuteBLayout>;
< 
<   using Epilogue = typename cutlass::epilogue::threadblock::symmetric::
<       DefaultEpilogueTensorOpDequant<
<           ThreadblockShape, typename DefaultGemm::Mma::Operator,
<           DefaultGemm::kPartitionsK, EpilogueOutputOp, EpilogueOutputOp::kCount,
<           ScatterD, PermuteDLayout>::Epilogue;
< 
<   using GemmKernel =
<       kernel::symmetric::GemmDequant<typename DefaultGemm::Mma, Epilogue,
<                                      ThreadblockSwizzle, SplitKSerial>;
< };
< ////////////////////////////////////////////////////////////////////////////////
< 
< }  // namespace symmetric
< }  // namespace kernel
< }  // namespace gemm
< }  // namespace cutlass
``` 
## 24. 文件/symmetric/gemm/kernel/gemm_dequant.h 的修改内容为：
``` 1,31d0
< /***************************************************************************************************
<  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
<  *reserved. SPDX-License-Identifier: BSD-3-Clause
<  *
<  * Redistribution and use in source and binary forms, with or without
<  * modification, are permitted provided that the following conditions are met:
<  *
<  * 1. Redistributions of source code must retain the above copyright notice,
<  *this list of conditions and the following disclaimer.
<  *
<  * 2. Redistributions in binary form must reproduce the above copyright notice,
<  * this list of conditions and the following disclaimer in the documentation
<  * and/or other materials provided with the distribution.
<  *
<  * 3. Neither the name of the copyright holder nor the names of its
<  * contributors may be used to endorse or promote products derived from
<  * this software without specific prior written permission.
<  *
<  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
<  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
<  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
<  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
<  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
<  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
<  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
<  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
<  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
<  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
<  *POSSIBILITY OF SUCH DAMAGE.
<  *
<  **************************************************************************************************/
33,391d1
< /*! \file
<     \brief Template for a pipelined GEMM kernel. Does not compute batching or
<    support split-K.
< */
< 
< #pragma once
< 
< #include "cutlass/arch/arch.h"
< #include "cutlass/cutlass.h"
< #include "cutlass/gemm/gemm.h"
< #include "cutlass/matrix_coord.h"
< #include "cutlass/semaphore.h"
< 
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< namespace cutlass {
< namespace gemm {
< namespace kernel {
< namespace symmetric {
< 
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
<           typename Epilogue_,            ///! Epilogue
<           typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
<           bool SplitKSerial  ///! If true, code supporting split-K via serial
<                              /// reduction is enabled.
<           >
< struct GemmDequant {
<   using Mma = Mma_;
<   using Epilogue = Epilogue_;
<   using OutputOp = typename Epilogue::OutputOp;
<   using ThreadblockSwizzle = ThreadblockSwizzle_;
<   static bool const kSplitKSerial = SplitKSerial;
< 
<   /// Warp count (concept: GemmShape)
<   using WarpCount = typename Mma::WarpCount;
<   static int const kThreadCount = 32 * WarpCount::kCount;
< 
<   /// Parameters structure
<   struct Params {
<     cutlass::gemm::GemmCoord problem_size;
<     cutlass::gemm::GemmCoord grid_tiled_shape;
<     int swizzle_log_tile;
<     typename Mma::IteratorA::Params params_A;
<     typename Mma::IteratorA::TensorRef ref_A;
<     typename Mma::IteratorB::Params params_B;
<     typename Mma::IteratorB::TensorRef ref_B;
<     typename Epilogue::OutputTileIterator::Params params_C;
<     typename Epilogue::OutputTileIterator::TensorRef ref_C;
<     typename Epilogue::OutputTileIterator::Params params_D;
<     typename Epilogue::OutputTileIterator::TensorRef ref_D;
<     typename Epilogue::RowVecIterator::Params params_row_vec;
<     typename Epilogue::RowVecIterator::TensorRef ref_row_vec;
<     typename Epilogue::ColVecIterator::Params params_col_vec;
<     typename Epilogue::ColVecIterator::TensorRef ref_col_vec;
<     typename OutputOp::Params output_op;
<     int *semaphore;
<     int gemm_k_size;
<     // For gather+scatter operations
<     int const *gather_A_indices;
<     int const *gather_B_indices;
<     int const *scatter_D_indices;
< 
<     //
<     // Methods
<     //
< 
<     CUTLASS_HOST_DEVICE
<     Params() : swizzle_log_tile(0), semaphore(0), gemm_k_size(0) {}
< 
<     CUTLASS_HOST_DEVICE
<     Params(cutlass::gemm::GemmCoord const &problem_size,
<            cutlass::gemm::GemmCoord const &grid_tiled_shape,
<            typename Mma::IteratorA::TensorRef ref_A,
<            typename Mma::IteratorB::TensorRef ref_B,
<            typename Epilogue::OutputTileIterator::TensorRef ref_C,
<            typename Epilogue::OutputTileIterator::TensorRef ref_D,
<            typename Epilogue::RowVecIterator::TensorRef ref_row_vec,
<            typename Epilogue::ColVecIterator::TensorRef ref_col_vec,
<            typename OutputOp::Params output_op = typename OutputOp::Params(),
<            int *workspace = nullptr, int const *gather_A_indices = nullptr,
<            int const *gather_B_indices = nullptr,
<            int const *scatter_D_indices = nullptr)
<         : problem_size(problem_size),
<           grid_tiled_shape(grid_tiled_shape),
<           swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
<           params_A(ref_A.layout()),
<           ref_A(ref_A),
<           params_B(ref_B.layout()),
<           ref_B(ref_B),
<           params_C(ref_C.layout()),
<           ref_C(ref_C),
<           params_D(ref_D.layout()),
<           ref_D(ref_D),
<           params_row_vec(ref_row_vec.layout()),
<           ref_row_vec(ref_row_vec),
<           params_col_vec(ref_col_vec.layout()),
<           ref_col_vec(ref_col_vec),
<           output_op(output_op),
<           gather_A_indices(gather_A_indices),
<           gather_B_indices(gather_B_indices),
<           scatter_D_indices(scatter_D_indices) {
<       int total_gemm_k_iterations =
<           (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
<       int gemm_k_iterations =
<           (total_gemm_k_iterations + grid_tiled_shape.k() - 1) /
<           grid_tiled_shape.k();
< 
<       gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
< 
<       semaphore = workspace;
<     }
<   };
< 
<   /// Shared memory storage structure
<   union SharedStorage {
<     typename Mma::SharedStorage main_loop;
<     typename Epilogue::SharedStorage epilogue;
<   };
< 
<   //
<   // Methods
<   //
< 
<   CUTLASS_HOST_DEVICE
<   GemmDequant() {}
< 
<   /// Determines whether kernel satisfies alignment
<   CUTLASS_HOST_DEVICE
<   static Status can_implement(
<       cutlass::gemm::GemmCoord const &problem_size,
<       typename Mma::IteratorA::TensorRef ref_A,
<       typename Mma::IteratorB::TensorRef ref_B,
<       typename Epilogue::OutputTileIterator::TensorRef ref_C,
<       typename Epilogue::OutputTileIterator::TensorRef ref_D,
<       typename Epilogue::RowVecIterator::TensorRef ref_row_vec,
<       typename Epilogue::ColVecIterator::TensorRef ref_col_vec) {
<     static int const kAlignmentA =
<         (platform::is_same<typename Mma::IteratorA::Layout,
<                            layout::ColumnMajorInterleaved<32>>::value)
<             ? 32
<         : (platform::is_same<typename Mma::IteratorA::Layout,
<                              layout::ColumnMajorInterleaved<64>>::value)
<             ? 64
<             : Mma::IteratorA::AccessType::kElements;
<     static int const kAlignmentB =
<         (platform::is_same<typename Mma::IteratorB::Layout,
<                            layout::RowMajorInterleaved<32>>::value)
<             ? 32
<         : (platform::is_same<typename Mma::IteratorB::Layout,
<                              layout::RowMajorInterleaved<64>>::value)
<             ? 64
<             : Mma::IteratorB::AccessType::kElements;
<     static int const kAlignmentC =
<         (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
<                            layout::ColumnMajorInterleaved<32>>::value)
<             ? 32
<         : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
<                              layout::ColumnMajorInterleaved<64>>::value)
<             ? 64
<             : Epilogue::OutputTileIterator::kElementsPerAccess;
< 
<     if (!TensorRef_aligned(ref_A, kAlignmentA)) {
<       return Status::kErrorMisalignedOperand;
<     }
< 
<     if (!TensorRef_aligned(ref_B, kAlignmentB)) {
<       return Status::kErrorMisalignedOperand;
<     }
< 
<     if (!TensorRef_aligned(ref_C, kAlignmentC)) {
<       return Status::kErrorMisalignedOperand;
<     }
< 
<     if (!TensorRef_aligned(ref_D, kAlignmentC)) {
<       return Status::kErrorMisalignedOperand;
<     }
< 
<     if (!TensorRef_aligned(ref_row_vec, kAlignmentC)) {
<       return Status::kErrorMisalignedOperand;
<     }
< 
<     if (!TensorRef_aligned(ref_col_vec, kAlignmentC)) {
<       return Status::kErrorMisalignedOperand;
<     }
< 
<     return Status::kSuccess;
<   }
< 
<   /// Executes one GEMM
<   CUTLASS_DEVICE
<   void operator()(Params const &params, SharedStorage &shared_storage) {
<     // Compute threadblock location
<     ThreadblockSwizzle threadblock_swizzle;
< 
<     cutlass::gemm::GemmCoord threadblock_tile_offset =
<         threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
< 
<     // Early exit if CTA is out of range
<     if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
<         params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
<       return;
<     }
< 
<     // Compute initial location in logical coordinates
<     cutlass::MatrixCoord tb_offset_A{
<         threadblock_tile_offset.m() * Mma::Shape::kM,
<         threadblock_tile_offset.k() * params.gemm_k_size,
<     };
< 
<     cutlass::MatrixCoord tb_offset_B{
<         threadblock_tile_offset.k() * params.gemm_k_size,
<         threadblock_tile_offset.n() * Mma::Shape::kN};
< 
<     // Problem size is a function of threadblock index in the K dimension
<     int problem_size_k =
<         min(params.problem_size.k(),
<             (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
< 
<     // Compute threadblock-scoped matrix multiply-add
<     int gemm_k_iterations =
<         (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) /
<         Mma::Shape::kK;
< 
<     // Compute position within threadblock
<     int thread_idx = threadIdx.x;
< 
<     // Construct iterators to A and B operands
<     typename Mma::IteratorA iterator_A(
<         params.params_A, params.ref_A.data(),
<         {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_A,
<         params.gather_A_indices);
< 
<     typename Mma::IteratorB iterator_B(
<         params.params_B, params.ref_B.data(),
<         {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_B,
<         params.gather_B_indices);
< 
<     // Broadcast the warp_id computed by lane 0 to ensure dependent code
<     // is compiled as warp-uniform.
<     int warp_idx = canonical_warp_idx_sync();
<     int lane_idx = threadIdx.x % 32;
< 
<     //
<     // Main loop
<     //
< 
<     // Construct thread-scoped matrix multiply
<     Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
< 
<     typename Mma::FragmentC accumulators;
< 
<     accumulators.clear();
< 
<     if (!kSplitKSerial || gemm_k_iterations > 0) {
<       // Compute threadblock-scoped matrix multiply-add
<       mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
<           accumulators);
<     }
< 
<     //
<     // Epilogue
<     //
< 
<     OutputOp output_op(params.output_op);
< 
<     //
<     // Masked tile iterators constructed from members
<     //
< 
<     threadblock_tile_offset =
<         threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
< 
<     // assume identity swizzle
<     MatrixCoord threadblock_offset(
<         threadblock_tile_offset.m() * Mma::Shape::kM,
<         threadblock_tile_offset.n() * Mma::Shape::kN);
< 
<     int block_idx = threadblock_tile_offset.m() +
<                     threadblock_tile_offset.n() * params.grid_tiled_shape.m();
< 
<     // Construct the semaphore.
<     Semaphore semaphore(params.semaphore + block_idx, thread_idx);
< 
<     // If performing a reduction via split-K, fetch the initial synchronization
<     if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
<       // Fetch the synchronization lock initially but do not block.
<       semaphore.fetch();
< 
<       // Indicate which position in a serial reduction the output operator is
<       // currently updating
<       output_op.set_k_partition(threadblock_tile_offset.k(),
<                                 params.grid_tiled_shape.k());
<     }
< 
<     // Tile iterator loading from source tensor.
<     typename Epilogue::OutputTileIterator iterator_C(
<         params.params_C, params.ref_C.data(), params.problem_size.mn(),
<         thread_idx, threadblock_offset, params.scatter_D_indices);
< 
<     // Tile iterator writing to destination tensor.
<     typename Epilogue::OutputTileIterator iterator_D(
<         params.params_D, params.ref_D.data(), params.problem_size.mn(),
<         thread_idx, threadblock_offset, params.scatter_D_indices);
< 
<     typename Epilogue::RowVecIterator iterator_row_vec(
<         params.params_row_vec, params.ref_row_vec.data(),
<         params.problem_size.mn(), thread_idx, threadblock_offset,
<         params.scatter_D_indices);
< 
<     typename Epilogue::ColVecIterator iterator_col_vec(
<         params.params_col_vec, params.ref_col_vec.data(),
<         params.problem_size.mn(), thread_idx, threadblock_offset,
<         params.scatter_D_indices);
< 
<     Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
< 
<     // Wait on the semaphore - this latency may have been covered by iterator
<     // construction
<     if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
<       // For subsequent threadblocks, the source matrix is held in the 'D'
<       // tensor.
<       if (threadblock_tile_offset.k()) {
<         iterator_C = iterator_D;
<       }
< 
<       semaphore.wait(threadblock_tile_offset.k());
<     }
< 
<     // Execute the epilogue operator to update the destination tensor.
<     epilogue(output_op, iterator_D, accumulators, iterator_C, iterator_row_vec,
<              iterator_col_vec);
< 
<     //
<     // Release the semaphore
<     //
< 
<     if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
<       int lock = 0;
<       if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {
<         // The final threadblock resets the semaphore for subsequent grids.
<         lock = 0;
<       } else {
<         // Otherwise, the semaphore is incremented
<         lock = threadblock_tile_offset.k() + 1;
<       }
< 
<       semaphore.release(lock);
<     }
<   }
< };
< 
< /////////////////////////////////////////////////////////////////////////////////////////////////
< 
< }  // namespace symmetric
< }  // namespace kernel
< }  // namespace gemm
< }  // namespace cutlass
``` 

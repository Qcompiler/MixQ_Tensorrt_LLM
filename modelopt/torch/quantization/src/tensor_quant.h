#include <torch/extension.h>

// TODO: add descriptions for functions
void fake_tensor_quant_cuda_inplace(at::Tensor, at::Tensor, int, bool, bool);
at::Tensor fake_tensor_quant_cuda(at::Tensor, at::Tensor, int, bool, bool);
at::Tensor fake_tensor_quant_with_axis_cuda(at::Tensor, at::Tensor, int, int, bool, bool);
float bits_to_bound(int, int);
at::Tensor fake_e4m3fy_cuda(at::Tensor inputs);

// Dequantizes data using NF4 quantization scheme and per-block scaling factors.
//
// Args:
//   quantized_data (torch::Tensor): Quantized data packed into uint8.
//   scales (torch::Tensor): Scaling factors for each data block.
//   block_size (int): Number of elements per data block.
//
// Returns:
//   at::Tensor: Dequantized data as bf16 tensor.
at::Tensor NF4_dequantize_cuda(torch::Tensor, torch::Tensor, int);

// Quantizes data using NF4 quantization scheme and per-block scaling factors.
//
// Args:
//   quantized_data (torch::Tensor): High precision data in fp32/fp16/bf16.
//   scales (torch::Tensor): Scaling factors for each data block.
//   block_size (int): Number of elements per data block.
//
// Returns:
//   at::Tensor: Quantized data packed in uint8.
at::Tensor NF4_quantize_cuda(torch::Tensor, torch::Tensor, int);

// Dequantizes data using INT4 quantization scheme and per-block scaling factors.
//
// Args:
//   quantized_data (torch::Tensor): Quantized data packed into uint8.
//   scales (torch::Tensor): Scaling factors for each data block.
//   block_size (int): Number of elements per data block.
//
// Returns:
//   at::Tensor: Dequantized data as the same dtype of scales.
at::Tensor INT4_dequantize_cuda(torch::Tensor, torch::Tensor, int);

// Quantizes data using INT4 quantization scheme and per-block scaling factors.
//
// Args:
//   quantized_data (torch::Tensor): High precision data in fp32/fp16/bf16.
//   scales (torch::Tensor): Scaling factors for each data block.
//   block_size (int): Number of elements per data block.
//
// Returns:
//   at::Tensor: Quantized data packed in uint8.
at::Tensor INT4_quantize_cuda(torch::Tensor, torch::Tensor, int);

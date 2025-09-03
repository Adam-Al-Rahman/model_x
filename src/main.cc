// Copyright 2025 MODEL_X

#include <iostream>

#include "torch/torch.h"

int main(int argc, char **argv) {
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "Device: CUDA" << std::endl;
    device = torch::Device(torch::kCUDA);
  } else {
    std::cout << "Device: CPU" << std::endl;
  }

#ifdef __APPLE__
  device = torch::Device(torch::kMPS);
#endif
}

// Copyright 2025 MODEL_X

#include <iostream>
#include "torch/torch.h"

int main() {
  std::cout << "Hello, World!" << std::endl;

  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}

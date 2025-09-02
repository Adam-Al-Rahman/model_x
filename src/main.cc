// Copyright 2025 MODEL_X

#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>

#include "torch/torch.h"
#include "tensorboard_logger.h"

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

  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;

  const auto start_time = std::chrono::system_clock::now().time_since_epoch().count();
  const auto logger_path = std::filesystem::path(argv[1]).replace_extension("tfevents." + std::to_string(start_time));

  if (!std::filesystem::exists(logger_path.parent_path()))
    std::filesystem::create_directories(logger_path.parent_path());

  TensorBoardLogger logger(logger_path);

  // Log scalars in a simple loop
  for (int step = 1; step <= 100; ++step) {
      double loss = 1.0f / step;  // fake "loss" decreasing over time
      double accuracy = step / 100.0f;

      logger.add_scalar("train/loss", step, loss);
      logger.add_scalar("train/accuracy", step, accuracy);
  }
}

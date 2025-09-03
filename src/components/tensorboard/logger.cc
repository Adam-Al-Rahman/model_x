// Copyright 2025 MODEL_X

#include "src/components/tensorboard/logger.h"

namespace model_x::src::component {

void log_data(const TensorBoardLogger &logger) {
  const auto start_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  const auto logger_path = std::filesystem::path(argv[1]).replace_extension(
      "tfevents." + std::to_string(start_time));

  if (!std::filesystem::exists(logger_path.parent_path()))
    std::filesystem::create_directories(logger_path.parent_path());

  TensorBoardLogger logger(logger_path);

  // DEMO: Log scalars in a simple loop
  for (int step = 1; step <= 100; ++step) {
    double loss = 1.0f / step;  // fake "loss" decreasing over time
    double accuracy = step / 100.0f;

    logger.add_scalar("train/loss", step, loss);
    logger.add_scalar("train/accuracy", step, accuracy);
  }
}

}  // namespace model_x::src::component

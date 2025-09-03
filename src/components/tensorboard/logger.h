// Copyright 2025 MODEL_X

#ifndef SRC_COMPONENTS_TENSORBOARD_LOGGER_H_
#define SRC_COMPONENTS_TENSORBOARD_LOGGER_H_

#include <chrono>
#include <filesystem>
#include <string>

#include "tensorboard_logger.h"

namespace model_x::src::component {
void log_data(const TensorBoardLogger &logger);
}  // namespace model_x::src::component

#endif  // SRC_COMPONENTS_TENSORBOARD_LOGGER_H_

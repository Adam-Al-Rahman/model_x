// Copyright 2025 MODEL_X

#ifndef SRC_COMPONENTS_TOKENIZERS_TOKENIZERS_H_
#define SRC_COMPONENTS_TOKENIZERS_TOKENIZERS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace model_x::src::component {
std::vector<std::uint32_t> tokenizers(const std::u8string& text,
                                      const std::string& tokenizer,
                                      std::size_t vocab_size = 256);
}  // namespace model_x::src::component

#endif  // SRC_COMPONENTS_TOKENIZERS_TOKENIZERS_H_

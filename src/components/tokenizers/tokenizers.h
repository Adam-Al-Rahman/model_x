// Copyright 2025 MODEL_X
#ifndef SRC_COMPONENTS_TOKENIZERS_TOKENIZERS_H_
#define SRC_COMPONENTS_TOKENIZERS_TOKENIZERS_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace model_x::src::component {

// Tokenize 'text' (UTF-8 bytes) using the named tokenizer (e.g. "byte_pair").
// The API accepts std::string_view for zero-copy calling convenience.
// Returns numeric token IDs.
std::vector<std::uint32_t> tokenizers(std::string_view text,
                                      const std::string& tokenizer,
                                      std::size_t vocab_size = 256);

}  // namespace model_x::src::component

#endif  // SRC_COMPONENTS_TOKENIZERS_TOKENIZERS_H_

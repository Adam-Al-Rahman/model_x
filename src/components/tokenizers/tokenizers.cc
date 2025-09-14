// Copyright 2025 MODEL_X

#include "src/components/tokenizers/tokenizers.h"

#include "src/components/tokenizers/byte_pair.h"

namespace model_x::src::component {

std::vector<std::uint32_t> tokenizers(const std::u8string& text,
                                      const std::string& tokenizer,
                                      std::size_t vocab_size) {
  std::vector<std::uint32_t> tokens(text.begin(), text.end());

  if (tokenizer == "byte_pair") {
    auto encode = byte_pair_enc(tokens, 270);
    for (auto val : encode) std::cout << val << ' ';
  }

  return tokens;  // default tokens
}

}  // namespace model_x::src::component

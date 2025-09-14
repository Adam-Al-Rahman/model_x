// Copyright 2025 MODEL_X

#ifndef SRC_COMPONENTS_TOKENIZERS_BYTE_PAIR_H_
#define SRC_COMPONENTS_TOKENIZERS_BYTE_PAIR_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"

namespace model_x::src::component {

// TODO(Adam-Al-Rahman): Symbol struct base token creation and pairing
absl::flat_hash_map<std::pair<std::uint32_t, std::uint32_t>, std::size_t,
                    absl::Hash<std::pair<std::uint32_t, std::uint32_t>>>
pair_freq(const std::vector<std::uint32_t>& tokens) {
  absl::flat_hash_map<std::pair<std::uint32_t, std::uint32_t>, std::size_t,
                      absl::Hash<std::pair<std::uint32_t, std::uint32_t>>>
      pair_count;

  if (tokens.size() < 2) return pair_count;

  for (std::size_t idx = 1; idx < tokens.size(); ++idx) {
    std::pair<std::uint32_t, std::uint32_t> p(tokens[idx - 1], tokens[idx]);
    ++pair_count[p];
  }

  return pair_count;
}

std::pair<std::uint32_t, std::uint32_t> get_max_pair(
    const absl::flat_hash_map<
        std::pair<std::uint32_t, std::uint32_t>, std::size_t,
        absl::Hash<std::pair<std::uint32_t, std::uint32_t>>>& pair_count) {
  std::pair<std::uint32_t, std::uint32_t> max_pair = {0, 0};
  std::size_t max_val = 0;

  for (const auto& [key, val] : pair_count) {
    if (val > max_val) {
      max_val = val;
      max_pair = key;
    }
  }

  return max_pair;
}

// TODO(Adam-Al-Rahman): priority_queue based merging
std::vector<std::uint32_t> merge(
    const std::vector<std::uint32_t>& tokens,
    const std::pair<std::uint32_t, std::uint32_t>& pair,
    std::size_t current_vocab_id) {
  std::vector<std::uint32_t> new_ids;

  std::size_t idx = 0;
  while (idx < tokens.size()) {
    if (idx < (tokens.size() - 1) && tokens[idx] == pair.first &&
        tokens[idx + 1] == pair.second) {
      new_ids.push_back(current_vocab_id);
      idx += 2;
    } else {
      new_ids.push_back(tokens[idx]);
      idx += 1;
    }
  }

  return new_ids;
}

constexpr std::size_t K_PAIR_SIZE = 256;
std::vector<std::uint32_t> bpe_train(std::vector<std::uint32_t> tokens,
                                     const std::size_t vocab_size) {
  std::size_t num_merges = vocab_size - K_PAIR_SIZE;

  absl::flat_hash_map<std::pair<std::uint32_t, std::uint32_t>, std::size_t,
                      absl::Hash<std::pair<std::uint32_t, std::uint32_t>>>
      merges;

  std::size_t pre_tokenization_size = tokens.size();

  std::size_t current_pair_size = K_PAIR_SIZE;
  for (std::size_t idx = 0; idx < num_merges; ++idx) {
    auto pair_count = pair_freq(tokens);
    auto max_pair = get_max_pair(pair_count);
    current_pair_size = K_PAIR_SIZE + idx;
    std::cout << "merging (" << max_pair.first << ", " << max_pair.second
              << ") new token " << current_pair_size << "\n";
    tokens = merge(tokens, max_pair, current_pair_size);
    merges[max_pair] = current_pair_size;
  }

  std::cout << static_cast<float>(pre_tokenization_size) / tokens.size()
            << std::endl;

  return tokens;
}

}  // namespace model_x::src::component

#endif  // SRC_COMPONENTS_TOKENIZERS_BYTE_PAIR_H_

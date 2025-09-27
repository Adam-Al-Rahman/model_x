// Copyright 2025 MODEL_X

#ifndef SRC_COMPONENTS_TOKENIZERS_BYTE_PAIR_H_
#define SRC_COMPONENTS_TOKENIZERS_BYTE_PAIR_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace model_x::src::component {

constexpr std::size_t K_PAIR_SIZE = 256;

struct PairKey {
  std::uint32_t first;
  std::uint32_t second;
  bool operator==(const PairKey &o) const noexcept {
    return first == o.first && second == o.second;
  }
};

struct PairKeyHash {
  std::size_t operator()(const PairKey &p) const noexcept {
    std::uint64_t v = (static_cast<std::uint64_t>(p.first) << 32) |
                      static_cast<std::uint64_t>(p.second);
    return static_cast<std::size_t>(v ^ (v >> 33));
  }
};

struct MergeInfo {
  std::uint32_t id;
  std::size_t rank;
};

class BPE {
 public:
  // Construct BPE with optional special token map (string -> id).
  // Special token IDs should be unique and preferably >= K_PAIR_SIZE.
  explicit BPE(
      const std::unordered_map<std::string, std::uint32_t> &special_tokens = {})
      : special_tokens_(special_tokens) {
    // ensure next_id_ doesn't collide with provided special token ids
    std::uint32_t max_special = 0;
    for (const auto &kv : special_tokens_) {
      if (kv.second > max_special) max_special = kv.second;
    }
    next_id_ = std::max<std::uint32_t>(static_cast<std::uint32_t>(K_PAIR_SIZE),
                                       max_special + 1);
  }

  // Train BPE merges from numeric tokens up to vocab_size (includes base 256).
  std::vector<std::uint32_t> train(std::vector<std::uint32_t> tokens,
                                   std::size_t vocab_size) {
    if (vocab_size <= K_PAIR_SIZE) return tokens;
    std::size_t num_merges = vocab_size - K_PAIR_SIZE;

    for (std::size_t i = 0; i < num_merges; ++i) {
      auto counts = pair_freq(tokens);
      if (counts.empty()) break;

      std::size_t max_count = 0;
      PairKey max_pair = get_max_pair(counts, max_count);
      if (max_count == 0) break;

      merges_.emplace(max_pair, MergeInfo{next_id_, i});
      // optional logging:
      // std::cout << "Merge (" << max_pair.first << ", " << max_pair.second <<
      // ") -> " << next_id_ << "\n";

      tokens = merge(tokens, max_pair, next_id_);
      ++next_id_;
    }
    return tokens;
  }

  // Encode numeric tokens using learned merges (used by train/inference
  // pipelines).
  std::vector<std::uint32_t> encode(
      const std::vector<std::uint32_t> &tokens) const {
    std::vector<std::uint32_t> t = tokens;
    if (t.size() < 2) return t;

    while (true) {
      bool found = false;
      std::size_t best_rank = static_cast<std::size_t>(-1);
      PairKey best_pair{0, 0};

      for (std::size_t i = 1; i < t.size(); ++i) {
        PairKey p{t[i - 1], t[i]};
        auto it = merges_.find(p);
        if (it != merges_.end() && it->second.rank < best_rank) {
          best_rank = it->second.rank;
          best_pair = p;
          found = true;
        }
      }

      if (!found) break;
      std::uint32_t new_id = merges_.at(best_pair).id;
      t = merge(t, best_pair, new_id);
    }

    return t;
  }

  // Encode strings -> numeric tokens, handling special tokens atomically.
  // Normal strings are converted byte-wise to uint8 values (0..255).
  std::vector<std::uint32_t> encode(
      const std::vector<std::string> &token_strs) const {
    std::vector<std::uint32_t> numeric;
    numeric.reserve(token_strs.size() * 4);

    for (const auto &s : token_strs) {
      auto sit = special_tokens_.find(s);
      if (sit != special_tokens_.end()) {
        // atomic special token ID
        numeric.push_back(sit->second);
      } else {
        // byte-wise encoding (UTF-8 bytes). If you prefer codepoints, replace
        // this block.
        for (unsigned char c : s)
          numeric.push_back(static_cast<std::uint32_t>(c));
      }
    }

    // apply BPE merges
    return encode(numeric);
  }

  // Decode numeric ids back into a string, restoring special tokens.
  // Special tokens are appended as their full string; other IDs are expanded
  // recursively into bytes.
  std::string decode(const std::vector<std::uint32_t> &ids) const {
    // build base vocab (0..255 -> single byte chars)
    std::array<std::string, K_PAIR_SIZE> vocab;
    for (std::uint16_t i = 0; i < K_PAIR_SIZE; ++i)
      vocab[i] = std::string(1, static_cast<char>(i));

    // reverse merges: id -> PairKey
    absl::flat_hash_map<std::uint32_t, PairKey> rev_merges;
    rev_merges.reserve(merges_.size());
    for (const auto &kv : merges_) rev_merges[kv.second.id] = kv.first;

    // reverse special tokens
    std::unordered_map<std::uint32_t, std::string> rev_specials;
    for (const auto &kv : special_tokens_) rev_specials[kv.second] = kv.first;

    std::string out;
    out.reserve(ids.size() * 2);

    for (auto id : ids) {
      // special token check first
      auto sit = rev_specials.find(id);
      if (sit != rev_specials.end()) {
        out += sit->second;
        continue;
      }

      // iterative expansion with stack for merged IDs
      std::stack<std::uint32_t> st;
      st.push(id);
      while (!st.empty()) {
        std::uint32_t cur = st.top();
        st.pop();

        // if it's a base byte
        if (cur < K_PAIR_SIZE) {
          out += vocab[cur];
          continue;
        }

        // if cur corresponds to a merged pair
        auto mit = rev_merges.find(cur);
        if (mit != rev_merges.end()) {
          // push second then first so first is processed first (stack LIFO)
          st.push(mit->second.second);
          st.push(mit->second.first);
          continue;
        }

        // unknown id -> replacement char
        out += "\uFFFD";
      }
    }

    return out;
  }

  // expose special tokens map
  const std::unordered_map<std::string, std::uint32_t> &special_tokens() const {
    return special_tokens_;
  }

 private:
  std::uint32_t next_id_{static_cast<std::uint32_t>(K_PAIR_SIZE)};
  absl::flat_hash_map<PairKey, MergeInfo, PairKeyHash> merges_;
  std::unordered_map<std::string, std::uint32_t> special_tokens_;

  static absl::flat_hash_map<PairKey, std::size_t, PairKeyHash> pair_freq(
      const std::vector<std::uint32_t> &tokens) {
    absl::flat_hash_map<PairKey, std::size_t, PairKeyHash> counts;
    if (tokens.size() < 2) return counts;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      PairKey p{tokens[i - 1], tokens[i]};
      ++counts[p];
    }
    return counts;
  }

  static PairKey get_max_pair(
      const absl::flat_hash_map<PairKey, std::size_t, PairKeyHash> &counts,
      std::size_t &out_max) {
    PairKey best{0, 0};
    out_max = 0;
    for (const auto &kv : counts) {
      if (kv.second > out_max) {
        out_max = kv.second;
        best = kv.first;
      }
    }
    return best;
  }

  static std::vector<std::uint32_t> merge(
      const std::vector<std::uint32_t> &tokens, const PairKey &pair,
      std::uint32_t new_id) {
    std::vector<std::uint32_t> out;
    out.reserve(tokens.size());
    std::size_t i = 0;
    while (i < tokens.size()) {
      if (i + 1 < tokens.size() && tokens[i] == pair.first &&
          tokens[i + 1] == pair.second) {
        out.push_back(new_id);
        i += 2;
      } else {
        out.push_back(tokens[i]);
        i += 1;
      }
    }
    return out;
  }
};

}  // namespace model_x::src::component

#endif  // SRC_COMPONENTS_TOKENIZERS_BYTE_PAIR_H_

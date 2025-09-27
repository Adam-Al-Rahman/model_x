// Copyright 2025 MODEL_X

#include "src/components/tokenizers/tokenizers.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "src/components/tokenizers/byte_pair.h"

namespace model_x::src::component {

// Helpers are private to this translation unit.
namespace {

// Convert std::string_view (bytes) -> std::u8string (char8_t) safely.
static std::u8string stringview_to_u8string(std::string_view sv) {
  std::u8string out;
  out.reserve(sv.size());
  for (unsigned char c : sv) {
    out.push_back(static_cast<char8_t>(c));
  }
  return out;
}

// Convert std::u8string -> std::string (byte-wise) without reinterpret_cast.
static std::string u8_to_string(const std::u8string& u8s) {
  std::string out;
  out.reserve(u8s.size());
  for (char8_t c : u8s) {
    out.push_back(static_cast<char>(static_cast<unsigned char>(c)));
  }
  return out;
}

// Prepare text with GPT-style special tokens: prefix + body + suffix.
static std::vector<std::string> prepare_with_specials(
    const std::u8string& body_u8) {
  const std::string prefix = "<|fim_prefix|>";
  const std::string suffix = "<|endoftext|>";
  std::vector<std::string> out;
  out.reserve(3);
  out.push_back(prefix);
  out.push_back(u8_to_string(body_u8));  // convert bytes -> std::string body
  out.push_back(suffix);
  return out;
}

}  // namespace

std::vector<std::uint32_t> tokenizers(std::string_view text,
                                      const std::string& tokenizer,
                                      std::size_t vocab_size) {
  // Standard special tokens (IDs should not collide with learned merges)
  const std::unordered_map<std::string, std::uint32_t> special_tokens = {
      {"<|endoftext|>", 50256},
      {"<|fim_prefix|>", 50257},
      {"<|fim_middle|>", 50258},
      {"<|fim_suffix|>", 50259},
  };

  // Convert input bytes -> u8string for internal operations
  std::u8string u8text = stringview_to_u8string(text);

  // Prepare vector<string> with special tokens inserted.
  std::vector<std::string> prepared = prepare_with_specials(u8text);

  if (tokenizer == "byte_pair") {
    // instantiate BPE with special tokens
    BPE bpe(special_tokens);

    // convert input to numeric bytes for training (byte-wise)
    std::vector<std::uint32_t> bytes;
    bytes.reserve(u8text.size());
    for (char8_t c : u8text) {
      bytes.push_back(
          static_cast<std::uint32_t>(static_cast<unsigned char>(c)));
    }

    // Train merges only if requested (vocab_size > base 256)
    if (vocab_size > K_PAIR_SIZE) {
      bpe.train(bytes, vocab_size);
    }

    // Encode the prepared vector<string> (special tokens handled atomically)
    std::vector<std::uint32_t> encoded = bpe.encode(prepared);

    // Optional: verify round-trip decoding (remove or guard in production)
    auto decoded = bpe.decode(encoded);
    std::cout << "Decoded (round-trip check): " << decoded << "\n";

    return encoded;
  }

  // Fallback: plain byte-wise encoding if unknown tokenizer
  std::vector<std::uint32_t> fallback;
  fallback.reserve(u8text.size());
  for (char8_t c : u8text) {
    fallback.push_back(
        static_cast<std::uint32_t>(static_cast<unsigned char>(c)));
  }
  return fallback;
}

}  // namespace model_x::src::component
